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

MM-Fi 17关节定义（COCO格式）：
  0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
  5:l_shoulder  6:r_shoulder  7:l_elbow  8:r_elbow
  9:l_wrist  10:r_wrist  11:l_hip  12:r_hip
  13:l_knee  14:r_knee  15:l_ankle  16:r_ankle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  人体骨骼连接对（用于 bone loss）
# ─────────────────────────────────────────────
HUMAN_BONES = [
    (0, 1), (0, 2),          # 鼻-眼
    (1, 3), (2, 4),          # 眼-耳
    (3, 5), (4, 6),          # 耳-肩
    (5, 6),                  # 两肩
    (5, 7), (7, 9),          # 左臂
    (6, 8), (8, 10),         # 右臂
    (5, 11), (6, 12),        # 躯干
    (11, 12),                # 髋
    (11, 13), (13, 15),      # 左腿
    (12, 14), (14, 16),      # 右腿
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

        # per-point MLP（共享权重，用 Conv1d 实现）
        self.conv1 = nn.Conv1d(3,          64,         1)
        self.conv2 = nn.Conv1d(64,         local_dim,  1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(local_dim)

        # 全局特征压缩
        self.fc_global = nn.Sequential(
            nn.Linear(local_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
        )

    def forward(self, x):
        """x: (B, N, 3)"""
        pts = x.transpose(1, 2)              # (B, 3, N)

        h = F.relu(self.bn1(self.conv1(pts)))  # (B, 64, N)
        h = F.relu(self.bn2(self.conv2(h)))    # (B, local_dim, N)

        f_local = h.transpose(1, 2)            # (B, N, local_dim)

        # MaxPool → global
        g = h.max(dim=-1)[0]                   # (B, local_dim)
        f_global = self.fc_global(g)            # (B, global_dim)

        return f_local, f_global


# ─────────────────────────────────────────────
#  Joint Feature Aggregator（Attention-based FFS）
# ─────────────────────────────────────────────
class JointFeatureAggregator(nn.Module):
    """
    用可学习关节嵌入作为 Query，f_local 作为 Key/Value，
    为每个关节在特征空间中软聚合最相关的局部点云证据。

    输入：
      f_local  (B, N, feat_dim)
    输出：
      f_joint  (B, num_joints, feat_dim)
    """
    def __init__(self, num_joints=17, feat_dim=128):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim   = feat_dim
        self.scale      = feat_dim ** -0.5

        # 可学习关节嵌入作为 Query
        self.joint_emb = nn.Embedding(num_joints, feat_dim)

        # Q / K / V 投影
        self.proj_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_o = nn.Linear(feat_dim, feat_dim)

    def forward(self, f_local):
        """
        f_local: (B, N, feat_dim)
        return:  (B, num_joints, feat_dim)
        """
        B, N, D = f_local.shape

        # 关节嵌入作为 Query: (num_joints, D) → (B, num_joints, D)
        idx = torch.arange(self.num_joints, device=f_local.device)
        q = self.joint_emb(idx).unsqueeze(0).expand(B, -1, -1)  # (B, J, D)
        q = self.proj_q(q)

        # 点云特征作为 Key / Value
        k = self.proj_k(f_local)   # (B, N, D)
        v = self.proj_v(f_local)   # (B, N, D)

        # Attention: (B, J, N)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale   # (B, J, N)
        attn = F.softmax(attn, dim=-1)

        # 加权聚合: (B, J, D)
        out = torch.bmm(attn, v)        # (B, J, D)
        out = self.proj_o(out)          # (B, J, D)

        return out


# ─────────────────────────────────────────────
#  残差 MLP：per-joint 坐标回归头
# ─────────────────────────────────────────────
class JointRegressorMLP(nn.Module):
    """
    输入：(B, J, in_dim)
    输出：(B, J, 3)  — 关节坐标
    参数在所有 J 个关节之间共享（Conv1d 实现）
    """
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
        # 残差捷径
        self.shortcut = nn.Conv1d(in_dim, 3, 1)

        # 输出层零初始化（使训练初期预测接近零修正量）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """x: (B, J, in_dim)"""
        h = x.transpose(1, 2)          # (B, in_dim, J)
        out = self.net(h) + self.shortcut(h)   # (B, 3, J)
        return out.transpose(1, 2)     # (B, J, 3)


# ─────────────────────────────────────────────
#  SkelKD 主模型
# ─────────────────────────────────────────────
class SkelKD(nn.Module):
    """
    从稀疏 mmWave 点云直接回归 17 个人体关节 3D 坐标。

    cfg 字段：
      num_joints   : int   = 17
      local_dim    : int   = 128
      global_dim   : int   = 256
    """
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
        """
        mmwave_pts: (B, N, 3)  稀疏毫米波点云
        return:     (B, 17, 3) 预测关节坐标
        """
        B, N, _ = mmwave_pts.shape

        # Step 1: 特征提取
        f_local, f_global = self.encoder(mmwave_pts)
        # f_local:  (B, N, 128)
        # f_global: (B, 256)

        # Step 2: 关节特征聚合（Attention-based FFS）
        f_joint = self.aggregator(f_local)           # (B, 17, 128)

        # Step 3: 拼接全局特征
        f_global_exp = f_global.unsqueeze(1).expand(-1, self.num_joints, -1)  # (B, 17, 256)
        f_cat = torch.cat([f_joint, f_global_exp], dim=-1)                   # (B, 17, 384)

        # Step 4: 坐标回归
        joints = self.regressor(f_cat)               # (B, 17, 3)

        return joints


# ─────────────────────────────────────────────
#  损失函数
# ─────────────────────────────────────────────
def mpjpe_loss(pred, gt):
    """
    Mean Per Joint Position Error
    pred, gt: (B, J, 3)
    return: scalar
    """
    return torch.norm(pred - gt, dim=-1).mean()


def bone_length_loss(pred, gt, bone_pairs=HUMAN_BONES):
    """
    骨骼长度一致性：预测骨骼长度与 GT 骨骼长度的 L1 误差
    pred, gt: (B, J, 3)
    return: scalar
    """
    loss = 0.
    for (i, j) in bone_pairs:
        pred_len = torch.norm(pred[:, i] - pred[:, j], dim=-1)  # (B,)
        gt_len   = torch.norm(gt[:, i]   - gt[:, j],   dim=-1)  # (B,)
        loss    += torch.abs(pred_len - gt_len).mean()
    return loss / len(bone_pairs)


def skelkd_loss(pred, gt, bone_weight=0.1):
    """
    总损失 = MPJPE + bone_weight * 骨骼长度损失
    pred, gt: (B, 17, 3)
    """
    m_loss = mpjpe_loss(pred, gt)
    b_loss = bone_length_loss(pred, gt)
    return m_loss + bone_weight * b_loss, m_loss, b_loss


# ─────────────────────────────────────────────
#  数据增强：训练时对 GT 骨骼施加扰动（验证用）
# ─────────────────────────────────────────────
def augment_skeleton(skeleton, noise_std=0.03, occlude_prob=0.3, max_occlude=4):
    """
    skeleton: (B, 17, 3)  GT 骨骼坐标
    return:   (B, 17, 3)  加噪 + 遮挡后的骨骼（用于对比实验，SkelKD 不需要骨骼输入）
    """
    s = skeleton.clone()
    B = s.shape[0]
    # 高斯噪声
    s = s + torch.randn_like(s) * noise_std
    # 随机遮挡
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
    print("pred shape:", pred.shape)          # (4, 17, 3)

    total, m, b = skelkd_loss(pred, skeleton_gt)
    print(f"loss={total:.4f}  mpjpe={m:.4f}  bone={b:.4f}")

    # 参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params / 1e6:.2f} M")