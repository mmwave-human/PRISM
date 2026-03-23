"""
model/scorenet/score.py
=======================
SKD-Net 条件扩散 Score Network

ConditionNet 双路条件（修正版）：
  ──────────────────────────────────────────────────────
  路径①  骨骼 GCN
    skeleton (B,17,3)
      → 2层 GCN（H36M-17 邻接矩阵）
      → (B, 17, gcn_C)  节点级特征
      → 全局池化 → Linear → (B, p_dim)   加到 t_emb（全局姿态调制）
      → Linear → (B, hidden, 17)         K/V 序列 ①

  路径②  粗糙点云几何
    coarse_pts (B, 128, 3)   ← GeoAE 输出，代替 raw mmWave
      → Conv1d 逐点特征提取
      → (B, hidden, 128)     K/V 序列 ②

  K/V = cat[①, ②] = (B, hidden, 145)
  ──────────────────────────────────────────────────────

  ★ 设计说明：
    raw mmWave 点云只作为 GeoAE 的输入，不直接进入 ConditionNet。
    coarse_pts 已包含了从稀疏 mmWave 中提炼出的可见区域几何信息，
    是 mmWave 几何条件的"精炼代理"，直接作为第②路 token 注入。

  condition dict 键（调用方传入）：
    'skeleton'  (B, 17,  3)   骨骼关节坐标（与点云坐标系对齐）
    'coarse'    (B, 128, 3)   GeoAE 输出的粗糙点云

  ★ 修正记录：
    - GCN 邻接矩阵从 COCO-17 修正为 H36M-17（与 MMFi ground_truth.npy 关节顺序一致）
    - 修复 global_cond 互斥 bug（label 和骨骼条件同时叠加）
    - skel_global 直接编码原始坐标，绕过 GCN 归一化
    - GCNLayer 使用 LayerNorm([17, C])，推理稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

from model.layers import TimeEmbedding, LabelEmbedding, ResidualBlock, FinalLayer
from tools.io import dict2namespace


# ──────────────────────────────────────────────────────────────
#  H36M-17 骨骼邻接矩阵（对称归一化）
# ──────────────────────────────────────────────────────────────

def _build_h36m17_adj() -> torch.Tensor:
    """
    构造 H36M-17 关节的对称归一化邻接矩阵 Â。
    Â = D^{-1/2} (A + I) D^{-1/2}，形状 (17, 17)。

    ★ 修正：MMFi ground_truth.npy 使用 H36M-17 格式，非 COCO-17。
    实测关节坐标验证：
      joint 3,6 (z最低) = R_Ankle, L_Ankle（双脚对称）
      joint 10  (z最高) = Head
      joint 0   (z≈0)   = Pelvis（骨盆中心）

    H36M-17 关节索引：
      0  Pelvis       ← 骨盆中心（根节点）
      1  R_Hip        2  R_Knee      3  R_Ankle
      4  L_Hip        5  L_Knee      6  L_Ankle
      7  Spine        8  Thorax      9  Neck        10 Head
      11 L_Shoulder   12 L_Elbow     13 L_Wrist
      14 R_Shoulder   15 R_Elbow     16 R_Wrist
    """
    edges = [
        (0, 1), (1, 2), (2, 3),            # 右腿：Pelvis → R_Hip → R_Knee → R_Ankle
        (0, 4), (4, 5), (5, 6),            # 左腿：Pelvis → L_Hip → L_Knee → L_Ankle
        (0, 7), (7, 8), (8, 9), (9, 10),   # 脊柱：Pelvis → Spine → Thorax → Neck → Head
        (8, 11), (11, 12), (12, 13),        # 左臂：Thorax → L_Shoulder → L_Elbow → L_Wrist
        (8, 14), (14, 15), (15, 16),        # 右臂：Thorax → R_Shoulder → R_Elbow → R_Wrist
    ]
    A = torch.zeros(17, 17)
    for i, j in edges:
        A[i, j] = 1.
        A[j, i] = 1.
    A = A + torch.eye(17)                          # 加自环
    D = A.sum(dim=1).pow(-0.5)                     # D^{-1/2}
    return D.unsqueeze(1) * A * D.unsqueeze(0)     # (17, 17)


# ──────────────────────────────────────────────────────────────
#  GCN 模块（路径①）
# ──────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    单层图卷积：H' = σ(Â @ H @ W)
    输入/输出：(B, 17, C)

    归一化策略：
      ★ 不用 LayerNorm(C)：对每个节点的 C 维特征归一化 → 所有节点范数相同 → 幅值信息丢失
      ★ 不用 BatchNorm1d(17)：eval 模式用 running stats，OOD 输入会爆炸（推理时 norm→156）
      ✓ 用 LayerNorm([17, C])（对整个 (17,C) 张量归一化）：
          - 保留节点间相对幅值差异（节点维不独立归一化）
          - 不依赖 running stats，推理 OOD 稳定
          - 等价于对 (B, 17*C) 做 LN，强度比逐节点 LN 弱
    """
    def __init__(self, in_channels: int, out_channels: int,
                 adj: torch.Tensor, act: bool = True):
        super().__init__()
        self.register_buffer('adj', adj)   # (17,17)，不参与梯度
        self.linear = nn.Linear(in_channels, out_channels)
        # ★ 对 (17, out_channels) 整体归一化，保留节点间幅值差异，推理稳定
        self.norm   = nn.LayerNorm([17, out_channels])
        self.act    = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 17, C_in) → (B, 17, C_out)"""
        x = self.adj @ x           # 图消息传播
        x = self.linear(x)
        x = self.norm(x)           # 对 (17, C) 整体归一化，不压缩节点间差异
        return F.gelu(x) if self.act else x


class SkeletonGCN(nn.Module):
    """
    2 层 GCN：骨骼关节坐标 → 节点级拓扑特征序列。

    输入：  skeleton (B, 17, 3)
    输出：  (B, 17, gcn_channels)

    ★ 修正：
      1. 邻接矩阵从 COCO-17 改为 H36M-17
      2. 去掉末尾 LayerNorm（避免级联归一化导致 global_cond 失效）
    """
    def __init__(self, gcn_channels: int = 256):
        super().__init__()
        adj = _build_h36m17_adj()       # ★ 修正：H36M-17
        mid = gcn_channels // 2
        self.input_proj = nn.Linear(3, mid)
        self.gcn1 = GCNLayer(mid,          gcn_channels, adj, act=True)
        self.gcn2 = GCNLayer(gcn_channels, gcn_channels, adj, act=False)

    def forward(self, skel: torch.Tensor) -> torch.Tensor:
        """skel: (B, 17, 3) → (B, 17, gcn_channels)"""
        x = F.gelu(self.input_proj(skel))
        x = self.gcn1(x)
        x = self.gcn2(x)
        return x


# ──────────────────────────────────────────────────────────────
#  粗糙点云编码器（路径②）
# ──────────────────────────────────────────────────────────────

class CoarseEncoder(nn.Module):
    """
    将 GeoAE 输出的粗糙点云编码为 K/V token 序列。

    输入：  coarse_pts (B, 128, 3)
    输出：  (B, hidden_size, 128)   直接作为 K/V 注入 Transformer
    """
    def __init__(self, hidden_size: int, gcn_channels: int = 256):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv1d(3,           gcn_channels, 1),
            nn.GELU(),
            nn.Conv1d(gcn_channels, gcn_channels, 1),
        )
        self.kv_proj = nn.Linear(gcn_channels, hidden_size)

    def forward(self, coarse_pts: torch.Tensor) -> torch.Tensor:
        """coarse_pts: (B, 128, 3) → (B, hidden_size, 128)"""
        feat = self.feat_proj(
            coarse_pts.transpose(1, 2))         # (B, gcn_channels, 128)
        feat = feat.transpose(1, 2)             # (B, 128, gcn_channels)
        feat = self.kv_proj(feat)               # (B, 128, hidden_size)
        return feat.transpose(1, 2)             # (B, hidden_size, 128)


# ──────────────────────────────────────────────────────────────
#  ConditionNet（双路条件）
# ──────────────────────────────────────────────────────────────

class ConditionNet(nn.Module):
    """
    双路条件融合，输出 (kv_tokens, global_cond)：

      kv_tokens:   (B, hidden_size, 145)
                   = cat[ 骨骼 GCN (17), coarse_pts (128) ], dim=-1

      global_cond: (B, p_dim)
                   = 骨骼全局池化 → Linear

    condition dict 期望键：
      'skeleton'  (B, 17,  3)
      'coarse'    (B, 128, 3)
    """

    def __init__(self,
                 hidden_size:        int,
                 p_dim:              int,
                 gcn_channels:       int  = 256,
                 skeleton_condition: bool = True,
                 coarse_condition:   bool = True):
        super().__init__()
        self.skeleton_condition = skeleton_condition
        self.coarse_condition   = coarse_condition

        # ── 路径① 骨骼 GCN ──────────────────────────────────
        if skeleton_condition:
            self.skeleton_gcn = SkeletonGCN(gcn_channels)
            self.skel_kv_proj = nn.Linear(gcn_channels, hidden_size)
            # ★ skel_global：直接编码原始坐标，绕过 GCN 归一化
            _skel_global_l1 = nn.Linear(17 * 3, p_dim)
            _skel_global_l2 = nn.Linear(p_dim, p_dim)
            nn.init.zeros_(_skel_global_l2.weight)
            nn.init.zeros_(_skel_global_l2.bias)
            self.skel_global  = nn.Sequential(
                _skel_global_l1,
                nn.GELU(),
                _skel_global_l2,
            )

        # ── 路径② 粗糙点云 ──────────────────────────────────
        if coarse_condition:
            self.coarse_encoder = CoarseEncoder(hidden_size, gcn_channels)

    def forward(self, condition: dict):
        """
        condition: {'skeleton': (B,17,3), 'coarse': (B,128,3)}
        返回：(kv_tokens, global_cond)
        """
        kv_parts    = []
        global_cond = 0.

        # ── 路径① ─────────────────────────────────────────
        if 'skeleton' in condition and self.skeleton_condition:
            skel = condition['skeleton'].float()
            skel_feat = self.skeleton_gcn(skel)              # (B, 17, gcn_C)
            global_cond = self.skel_global(
                skel.reshape(skel.shape[0], -1))             # (B,51)→(B,p_dim)
            skel_kv = self.skel_kv_proj(
                skel_feat).transpose(1, 2)                   # (B, hidden, 17)

            # ── 诊断打印 ──────────────────────────────────────
            if not hasattr(self, '_diag_count'):
                self._diag_count = 0
            if self._diag_count % 200 == 0:
                gc_std = global_cond.std(dim=0).mean().item()
                gc_norm = global_cond.norm(dim=-1).mean().item()
                print(f"[DIAG step={self._diag_count}]"
                      f"  skel_kv norm={skel_kv.norm(dim=1).mean().item():.4f}"
                      f"  global_cond norm={gc_norm:.4f}  std={gc_std:.4f}"
                      f"  skel_input=[{skel.min().item():.3f},{skel.max().item():.3f}]")
            self._diag_count += 1

            kv_parts.append(skel_kv)

        # ── 路径② ─────────────────────────────────────────
        if 'coarse' in condition and self.coarse_condition:
            coarse_kv = self.coarse_encoder(
                condition['coarse'].float())                 # (B, hidden, 128)
            kv_parts.append(coarse_kv)

        # ── 拼接 ──────────────────────────────────────────
        if kv_parts:
            kv_tokens = torch.cat(kv_parts, dim=2)
        else:
            kv_tokens = 0.

        return kv_tokens, global_cond


# ──────────────────────────────────────────────────────────────
#  Score Network（主干不变）
# ──────────────────────────────────────────────────────────────

class Score(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg         = cfg
        self.z_dim       = cfg.z_dim
        self.out_dim     = self.z_dim
        self.z_scale     = cfg.z_scale
        self.hidden_size = cfg.hidden_size
        self.num_heads   = cfg.num_heads
        self.condition   = cfg.condition
        self.num_steps   = cfg.num_steps
        self.norm        = cfg.norm
        self.t_dim       = cfg.t_dim
        self.num_blocks  = cfg.num_blocks
        self.dropout     = cfg.dropout
        self.learn_sigma = cfg.learn_sigma
        self.unet        = cfg.unet
        self.AdaLN       = cfg.AdaLN

        if self.condition:
            self.c_net = ConditionNet(
                hidden_size        = self.hidden_size,
                p_dim              = self.t_dim,
                gcn_channels       = 256,
                skeleton_condition = True,
                coarse_condition   = getattr(cfg, 'coarse_condition', True),
            )

        if self.unet:
            self.Transformer_Up = nn.ModuleList([
                ResidualBlock(self.hidden_size, self.hidden_size,
                              self.t_dim, self.num_heads,
                              norm=self.norm, dropout_att=self.dropout,
                              dropout_mlp=self.dropout, act=cfg.act,
                              AdaLN=self.AdaLN)
                for _ in range(self.num_blocks // 2)])
            self.Transformer_Mid = ResidualBlock(
                self.hidden_size, self.hidden_size,
                self.t_dim, self.num_heads,
                norm=self.norm, dropout_att=self.dropout,
                dropout_mlp=self.dropout, act=cfg.act, AdaLN=self.AdaLN)
            self.Transformer_Down = nn.ModuleList([
                ResidualBlock(self.hidden_size * 2, self.hidden_size * 2,
                              self.t_dim, self.num_heads,
                              norm=self.norm, dropout_att=self.dropout,
                              dropout_mlp=self.dropout,
                              dim_out=self.hidden_size, act=cfg.act,
                              AdaLN=self.AdaLN)
                for _ in range(self.num_blocks // 2)])
        else:
            self.Transformer = nn.ModuleList([
                ResidualBlock(self.hidden_size, self.hidden_size,
                              self.t_dim, self.num_heads,
                              norm=self.norm, dropout_att=self.dropout,
                              dropout_mlp=self.dropout, act=cfg.act,
                              AdaLN=self.AdaLN)
                for _ in range(self.num_blocks)])

        if cfg.num_categorys > 1:
            self.LabelEmbedding = LabelEmbedding(
                cfg.num_categorys, self.t_dim, self.t_dim)
        else:
            self.label_dim = None

        self.ln_in         = nn.Conv1d(self.z_dim, self.hidden_size, 1)
        self.TimeEmbedding = TimeEmbedding(self.t_dim // 4, self.t_dim)
        self.ln_out        = FinalLayer(self.hidden_size, self.z_dim,
                                        self.t_dim, self.norm)

    def forward(self, x, t, label=None, condition=None):
        l_emb = self.LabelEmbedding(label) if label is not None else None

        if condition is not None and isinstance(condition, dict):
            condition = self.c_net(condition)
        else:
            condition = (None, 0.)

        t_emb = self.TimeEmbedding(t)
        # ★ 修复：label 和骨骼 global_cond 同时叠加
        c = t_emb
        if l_emb is not None:
            c = c + l_emb
        c = c + condition[1]

        x = x.transpose(1, 2)
        x = self.ln_in(x)

        if self.unet:
            x_list = [x]
            for layer in self.Transformer_Up:
                x = layer(x, condition[0], c)
                x_list.append(x)
            x = self.Transformer_Mid(x, condition[0], c)
            for layer in self.Transformer_Down:
                x = torch.cat((x, x_list.pop()), dim=1)
                x = layer(x, condition[0], c)
        else:
            for idx, layer in enumerate(self.Transformer):
                x = layer(x, condition[0] if idx % 2 == 0 else None, c)

        out = self.ln_out(x, c).transpose(1, 2)
        return out


# ──────────────────────────────────────────────────────────────
#  快速自测
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    def get_config(path):
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return dict2namespace(cfg)

    B = 4
    skel   = torch.randn(B, 17,  3).cuda()
    coarse = torch.randn(B, 128, 3).cuda()
    t      = torch.rand(B).cuda()
    z      = torch.randn(B, 8, 120).cuda()

    cfg   = get_config("experiments/Latent_Diffusion_Trainer/config.yaml")
    model = Score(cfg.score).cuda()

    condition = {'skeleton': skel, 'coarse': coarse}
    out = model(z, t, condition=condition)
    print("output shape  :", out.shape)

    kv, g = model.c_net(condition)
    print("kv_tokens     :", kv.shape)
    print("global_cond   :", g.shape)

    total = sum(p.numel() for p in model.parameters())
    print(f"Score 总参数量 : {total / 1e6:.2f} M")