"""
model/SkelKD/Network.py

SkelKD: Skeleton Keypoint Detector
从稀疏 mmWave 点云直接回归 17 个人体关节的 3D 坐标。

架构：
  mmWave X (B, N, 3)
      │
      ▼
  PCN Encoder
    per-point MLP: 3 → 64 → 128    → f_local  (B, N, 128)
    MaxPool + MLP:  128 → 256       → f_global (B, 256)
      │
      ▼
  Joint Feature Aggregator (Attention-based FFS)
    可学习关节嵌入 (17, 128) 作为 Query
    f_local 作为 Key / Value
    → f_joint (B, 17, 128)  每个关节从点云中软聚合最相关局部证据
      │
      ▼
  拼接 [f_joint(128) | f_global_expand(256)] → (B, 17, 384)
      │
      ▼
  残差 MLP (per-joint, 参数共享)
    384 → 256 → ReLU → 128 → ReLU → 3
    → delta_P (B, 17, 3)
      │
      ▼
  P_joint (B, 17, 3)

训练：
  - 输入增强：骨骼 GT 加噪 + 随机遮挡（模拟标定误差与遮挡）
  - 损失：MPJPE + 骨骼长度一致性
  - 监督：MM-Fi 提供的 skeleton GT

★ 修正：MM-Fi ground_truth.npy 使用 H36M-17 格式（非 COCO-17）

H36M-17 关节定义：
  0:Pelvis  1:R_Hip  2:R_Knee  3:R_Ankle
  4:L_Hip   5:L_Knee  6:L_Ankle
  7:Spine   8:Thorax  9:Neck  10:Head
  11:L_Shoulder  12:L_Elbow  13:L_Wrist
  14:R_Shoulder  15:R_Elbow  16:R_Wrist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  人体骨骼连接对（用于 bone loss）
#  ★ 修正：从 COCO-17 改为 H36M-17
# ─────────────────────────────────────────────
HUMAN_BONES = [
    (0, 1), (1, 2), (2, 3),            # 右腿：Pelvis → R_Hip → R_Knee → R_Ankle
    (0, 4), (4, 5), (5, 6),            # 左腿：Pelvis → L_Hip → L_Knee → L_Ankle
    (0, 7), (7, 8), (8, 9), (9, 10),   # 脊柱：Pelvis → Spine → Thorax → Neck → Head
    (8, 11), (11, 12), (12, 13),        # 左臂：Thorax → L_Shoulder → L_Elbow → L_Wrist
    (8, 14), (14, 15), (15, 16),        # 右臂：Thorax → R_Shoulder → R_Elbow → R_Wrist
]


# ─────────────────────────────────────────────
#  PCN Encoder：逐点特征提取
# ─────────────────────────────────────────────
class PCNEncoder(nn.Module):
    """
    输入：(B, N, 3)
    输出：
      f_local  (B, N, local_dim)   逐点局部特征
      f_global (B, global_dim)     全局特征
    """
    def __init__(self, local_dim=128, global_dim=256):
        super().__init__()
        self.local_dim  = local_dim
        self.global_dim = global_dim

        self.conv1 = nn.Conv1d(3,          64,         1)
        self.conv2 = nn.Conv1d(64,         local_dim,  1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(local_dim)

        self.fc_global = nn.Sequential(
            nn.Linear(local_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
        )

    def forward(self, x):
        """x: (B, N, 3)"""
        pts = x.transpose(1, 2)

        h = F.relu(self.bn1(self.conv1(pts)))
        h = F.relu(self.bn2(self.conv2(h)))

        f_local = h.transpose(1, 2)

        g = h.max(dim=-1)[0]
        f_global = self.fc_global(g)

        return f_local, f_global


# ─────────────────────────────────────────────
#  Joint Feature Aggregator（Attention-based FFS）
# ─────────────────────────────────────────────
class JointFeatureAggregator(nn.Module):
    """
    用可学习关节嵌入作为 Query，f_local 作为 Key/Value，
    为每个关节在特征空间中软聚合最相关的局部点云证据。
    """
    def __init__(self, num_joints=17, feat_dim=128):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim   = feat_dim
        self.scale      = feat_dim ** -0.5

        self.joint_emb = nn.Embedding(num_joints, feat_dim)

        self.proj_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_o = nn.Linear(feat_dim, feat_dim)

    def forward(self, f_local):
        B, N, D = f_local.shape

        idx = torch.arange(self.num_joints, device=f_local.device)
        q = self.joint_emb(idx).unsqueeze(0).expand(B, -1, -1)
        q = self.proj_q(q)

        k = self.proj_k(f_local)
        v = self.proj_v(f_local)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = self.proj_o(out)

        return out


# ─────────────────────────────────────────────
#  残差 MLP：per-joint 坐标回归头
# ─────────────────────────────────────────────
class JointRegressorMLP(nn.Module):
    def __init__(self, in_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1),
        )
        self.shortcut = nn.Conv1d(in_dim, 3, 1)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        h = x.transpose(1, 2)
        out = self.net(h) + self.shortcut(h)
        return out.transpose(1, 2)


# ─────────────────────────────────────────────
#  SkelKD 主模型
# ─────────────────────────────────────────────
class SkelKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_joints  = getattr(cfg, 'num_joints',  17)
        local_dim   = getattr(cfg, 'local_dim',  128)
        global_dim  = getattr(cfg, 'global_dim', 256)

        self.num_joints = num_joints

        self.encoder    = PCNEncoder(local_dim=local_dim, global_dim=global_dim)
        self.aggregator = JointFeatureAggregator(num_joints=num_joints, feat_dim=local_dim)
        self.regressor  = JointRegressorMLP(in_dim=local_dim + global_dim)

    def forward(self, mmwave_pts):
        B, N, _ = mmwave_pts.shape

        f_local, f_global = self.encoder(mmwave_pts)

        f_joint = self.aggregator(f_local)

        f_global_exp = f_global.unsqueeze(1).expand(-1, self.num_joints, -1)
        f_cat = torch.cat([f_joint, f_global_exp], dim=-1)

        joints = self.regressor(f_cat)

        return joints


# ─────────────────────────────────────────────
#  损失函数
# ─────────────────────────────────────────────
def mpjpe_loss(pred, gt):
    return torch.norm(pred - gt, dim=-1).mean()


def bone_length_loss(pred, gt, bone_pairs=HUMAN_BONES):
    loss = 0.
    for (i, j) in bone_pairs:
        pred_len = torch.norm(pred[:, i] - pred[:, j], dim=-1)
        gt_len   = torch.norm(gt[:, i]   - gt[:, j],   dim=-1)
        loss    += torch.abs(pred_len - gt_len).mean()
    return loss / len(bone_pairs)


def skelkd_loss(pred, gt, bone_weight=0.1):
    m_loss = mpjpe_loss(pred, gt)
    b_loss = bone_length_loss(pred, gt)
    return m_loss + bone_weight * b_loss, m_loss, b_loss


# ─────────────────────────────────────────────
#  数据增强
# ─────────────────────────────────────────────
def augment_skeleton(skeleton, noise_std=0.03, occlude_prob=0.3, max_occlude=4):
    s = skeleton.clone()
    B = s.shape[0]
    s = s + torch.randn_like(s) * noise_std
    for b in range(B):
        if torch.rand(1).item() < occlude_prob:
            n = torch.randint(1, max_occlude + 1, (1,)).item()
            idx = torch.randperm(17)[:n]
            s[b, idx] = 0.0
    return s


# ─────────────────────────────────────────────
#  单元测试
# ─────────────────────────────────────────────
if __name__ == '__main__':
    from types import SimpleNamespace

    cfg = SimpleNamespace(num_joints=17, local_dim=128, global_dim=256)
    model = SkelKD(cfg).cuda()

    B, N = 4, 100
    mmwave = torch.randn(B, N, 3).cuda()
    skeleton_gt = torch.randn(B, 17, 3).cuda()

    pred = model(mmwave)
    print("pred shape:", pred.shape)

    total, m, b = skelkd_loss(pred, skeleton_gt)
    print(f"loss={total:.4f}  mpjpe={m:.4f}  bone={b:.4f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params / 1e6:.2f} M")