"""
model/AutoEncoderE/Network.py
==============================
Auto-encoder E：从稀疏 mmWave 点云提取几何条件信号。

灵感来源：LAKe-Net (CVPR 2022) 的 Auto-encoder E 设计
  PCN Encoder：两级 Conv1d，提取逐点特征与全局特征
  SeedGenerator：ConvTranspose1d 将全局特征解卷积为粗糙点云

与 LAKe-Net 原版的差异：
  1. 输入 mmWave 50~256 点（原版为 2048 点），output_size 降为 256
  2. 去掉 VAE 重参数化（不需要 eps，直接输出点坐标）
  3. num_coarse = 128（原版 256），与 mmWave 输入量级匹配，覆盖全身轮廓
  4. 返回 coarse_pts + global_feat，供 ConditionNet 使用

设计说明：
  128 点的选取依据：
    - mmWave 输入约 50~150 点，128 > 输入点数，能覆盖遮挡区域
    - 128 < LiDAR GT 256 点，维持"粗糙→精细"的层次感
    - KV 序列 = 17(骨骼) + 128(coarse) = 145，注意力分布合理

数据流：
  mmwave (B, N, 3) → PCN_Encoder → global_feat (B, 256)
                   → SeedGenerator → coarse_pts (B, 128, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  基础模块（直接从 LAKe-Net 移植，零外部依赖）
# ─────────────────────────────────────────────────────────────

class MLP_Res(nn.Module):
    """
    残差 MLP 块（Conv1d 实现，与 LAKe-Net 保持一致）
    输入/输出：(B, C, N)
    """
    def __init__(self, in_dim: int = 128,
                 hidden_dim: int = None,
                 out_dim: int = 128):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1       = nn.Conv1d(in_dim,     hidden_dim, 1)
        self.conv_2       = nn.Conv1d(hidden_dim, out_dim,    1)
        self.conv_shortcut = nn.Conv1d(in_dim,    out_dim,    1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N)"""
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(F.relu(self.conv_1(x))) + shortcut
        return out


class PCN_Encoder(nn.Module):
    """
    PCN 双级特征提取器。

    第一级：逐点 Conv1d → 256-d 特征，MaxPool 得全局 256-d 特征
    第二级：逐点拼接全局特征后再升维 → output_size 特征

    输入：  x (B, 3, N)   ← Conv1d 格式
    输出：
      point_feat  (B, output_size, N)  逐点特征（含全局上下文）
      global_feat (B, output_size)     全局特征
    """
    def __init__(self, output_size: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(3,   128,         1)
        self.conv2 = nn.Conv1d(128, 256,         1)
        self.conv3 = nn.Conv1d(512, 512,         1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, 3, N)"""
        B, _, N = x.shape

        x1 = F.relu(self.conv1(x))                        # (B, 128, N)
        x2 = self.conv2(x1)                               # (B, 256, N)

        g1, _ = torch.max(x2, dim=2)                      # (B, 256) 第一级全局

        x3 = torch.cat([x2,
                        g1.unsqueeze(-1).expand(-1, -1, N)],
                       dim=1)                             # (B, 512, N)
        x4 = F.relu(self.conv3(x3))                       # (B, 512, N)
        x5 = self.conv4(x4)                               # (B, output_size, N)

        global_feat, _ = torch.max(x5, dim=2)             # (B, output_size)
        return x5, global_feat


class SeedGenerator(nn.Module):
    """
    种子点生成器：从全局特征解卷积出粗糙点云。

    核心思路（来自 LAKe-Net）：
      ConvTranspose1d(dim_feat, 128, num_pc)
      → 一步将全局标量特征"展开"为 num_pc 个 128-d 种子向量
      → 残差 MLP 精炼 × 3 → 3D 坐标输出

    输入：  feat (B, dim_feat)
    输出：  coarse_pts (B, num_pc, 3)
    """
    def __init__(self, dim_feat: int = 256, num_pc: int = 128):
        super().__init__()
        self.ps    = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128,            hidden_dim=64,  out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64,  3,  1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, dim_feat) → (B, num_pc, 3)"""
        f  = feat.unsqueeze(-1)                                    # (B, dim_feat, 1)
        x1 = self.ps(f)                                            # (B, 128, num_pc)
        x1 = self.mlp_1(
            torch.cat([x1, f.expand(-1, -1, x1.size(2))], dim=1)) # (B, 128, num_pc)
        x2 = self.mlp_2(x1)                                        # (B, 128, num_pc)
        x3 = self.mlp_3(
            torch.cat([x2, f.expand(-1, -1, x2.size(2))], dim=1)) # (B, 128, num_pc)
        pts = self.mlp_4(x3)                                       # (B, 3, num_pc)
        return pts.transpose(1, 2).contiguous()                    # (B, num_pc, 3)


# ─────────────────────────────────────────────────────────────
#  AutoEncoderE：对外接口
# ─────────────────────────────────────────────────────────────

class AutoEncoderE(nn.Module):
    """
    Auto-encoder E：mmWave 稀疏点云 → 粗糙几何条件。

    输入：  mmwave (B, N, 3)   N ≈ 50~256（mmWave 稀疏点云）
    输出：
      coarse_pts  (B, 128, 3)   粗糙点云，作为 ConditionNet 第②路 K/V
      global_feat (B, 256)      全局特征（可选用于特征对齐 loss）

    训练策略：在扩散训练（阶段②）中联合训练，不需要单独预训练。
    辅助监督：CD(coarse_pts, LiDAR_GT) × coarse_loss_weight
    """

    def __init__(self, cfg=None,
                 feat_dim:   int = 256,
                 num_coarse: int = 128):
        super().__init__()
        if cfg is not None:
            feat_dim   = getattr(cfg, 'feat_dim',   feat_dim)
            num_coarse = getattr(cfg, 'num_coarse', num_coarse)

        self.feat_dim   = feat_dim
        self.num_coarse = num_coarse

        self.encoder = PCN_Encoder(output_size=feat_dim)
        self.decoder = SeedGenerator(dim_feat=feat_dim, num_pc=num_coarse)

    def forward(self, mmwave: torch.Tensor):
        """
        mmwave: (B, N, 3)
        返回：coarse_pts (B, num_coarse, 3), global_feat (B, feat_dim)
        """
        x = mmwave.transpose(1, 2).contiguous()     # (B, 3, N)
        _, global_feat = self.encoder(x)            # (B, feat_dim)
        coarse_pts     = self.decoder(global_feat)  # (B, num_coarse, 3)
        return coarse_pts, global_feat


# ─────────────────────────────────────────────────────────────
#  快速自测
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, N = 4, 80
    mm = torch.randn(B, N, 3)

    model = AutoEncoderE(feat_dim=256, num_coarse=128)
    coarse, gfeat = model(mm)

    print(f"mmwave input  : {mm.shape}")      # (4, 80, 3)
    print(f"coarse_pts    : {coarse.shape}")  # (4, 128, 3)
    print(f"global_feat   : {gfeat.shape}")   # (4, 256)

    total = sum(p.numel() for p in model.parameters())
    print(f"参数量：{total / 1e6:.2f} M")