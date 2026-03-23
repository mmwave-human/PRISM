"""
pointr.py
=========
PoinTr adapted for MM-Fi dataset.
"PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers"
Yu et al., ICCV 2021.

Input : sparse mmWave point cloud (B, M, 3)
Output: coarse (B, num_queries, 3)  +  fine (B, num_fine, 3)

Architecture:
  1. PointTokenizer  : FPS + KNN grouping → proxy tokens
  2. Transformer Enc : refine input tokens
  3. Transformer Dec : num_queries learned tokens attend to encoder memory
  4. FoldingGenerator: each query token → num_per_token fine points
"""

import os
import sys
import torch
import torch.nn as nn

# FPS from compiled pointnet2 (installed via mmPoint)
_pn2_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../../mmPoint/utils/Pointnet2.PyTorch/pointnet2'))
sys.path.insert(0, _pn2_dir)
import pointnet2_utils as pn2


# ── KNN grouping (pure PyTorch, N=512 is small enough) ────────────────────────

def knn_group(xyz, num_centers, k):
    """
    FPS + KNN: sample num_centers centers, group k neighbors each.

    Args:
        xyz       : (B, N, 3)
        num_centers: int
        k         : int
    Returns:
        centers   : (B, num_centers, 3)
        grouped   : (B, num_centers, k, 3)  local coords relative to center
    """
    B, N, _ = xyz.shape

    # FPS
    fps_idx = pn2.furthest_point_sample(xyz.contiguous(), num_centers)      # (B, Nc)
    centers = pn2.gather_operation(
        xyz.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()                                           # (B, Nc, 3)

    # KNN: pairwise squared distances between centers and all points
    c  = centers.unsqueeze(2)       # (B, Nc, 1,  3)
    p  = xyz.unsqueeze(1)           # (B,  1, N,  3)
    dist = ((c - p) ** 2).sum(-1)  # (B, Nc, N)
    _, idx = dist.topk(k, dim=2, largest=False)  # (B, Nc, k)

    # Gather neighbours and compute local coordinates
    idx_flat = idx.reshape(B, -1)                        # (B, Nc*k)
    neighbours = torch.gather(
        xyz, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)
    ).reshape(B, num_centers, k, 3)                      # (B, Nc, k, 3)
    grouped = neighbours - centers.unsqueeze(2)          # local coords

    return centers, grouped


# ── Building blocks ────────────────────────────────────────────────────────────

class MiniPointNet(nn.Module):
    """Per-group local feature extractor: (B, M, k, 3) → (B, M, feat_dim)"""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(3,  64,       1), nn.BatchNorm2d(64),       nn.ReLU(),
            nn.Conv2d(64, 128,      1), nn.BatchNorm2d(128),      nn.ReLU(),
            nn.Conv2d(128, feat_dim, 1),
        )

    def forward(self, x):
        # x: (B, M, k, 3)
        x = x.permute(0, 3, 1, 2)       # (B, 3, M, k)
        x = self.mlp(x)                  # (B, feat_dim, M, k)
        return x.max(dim=3)[0].transpose(1, 2)  # (B, M, feat_dim)


class PointTokenizer(nn.Module):
    """Convert raw point cloud to geometry-aware proxy tokens."""

    def __init__(self, num_centers: int, k: int, feat_dim: int):
        super().__init__()
        self.num_centers = num_centers
        self.k           = k
        self.local_enc   = MiniPointNet(feat_dim)
        self.pos_emb     = nn.Linear(3, feat_dim)

    def forward(self, xyz):
        centers, grouped = knn_group(xyz, self.num_centers, self.k)
        tokens  = self.local_enc(grouped)       # (B, Nc, feat_dim)
        pos     = self.pos_emb(centers)         # (B, Nc, feat_dim)
        return centers, tokens + pos


class FoldingGenerator(nn.Module):
    """
    Generate num_per_token fine points per query via 2D grid folding.
    num_per_token must be a perfect square.
    """

    def __init__(self, feat_dim: int, num_per_token: int):
        super().__init__()
        gs = int(num_per_token ** 0.5)
        assert gs * gs == num_per_token, \
            f"num_per_token must be a perfect square, got {num_per_token}"
        self.num_per_token = num_per_token

        a   = torch.linspace(-0.1, 0.1, steps=gs)
        gx, gy = torch.meshgrid(a, a)
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (n, 2)
        self.register_buffer('grid', grid)

        self.coarse_head = nn.Linear(feat_dim, 3)
        self.fold_mlp = nn.Sequential(
            nn.Conv1d(feat_dim + 2, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,          64,  1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,           3,   1),
        )

    def forward(self, tokens):
        """tokens: (B, Q, feat_dim) → coarse (B,Q,3), fine (B,Q*n,3)"""
        B, Q, C = tokens.shape
        n = self.num_per_token

        coarse = self.coarse_head(tokens)                              # (B, Q, 3)

        feat_exp = tokens.unsqueeze(2).expand(B, Q, n, C)
        grid_exp = self.grid.unsqueeze(0).unsqueeze(0).expand(B, Q, n, 2)

        x    = torch.cat([feat_exp, grid_exp], dim=-1)                # (B, Q, n, C+2)
        x    = x.reshape(B * Q, n, C + 2).transpose(1, 2)            # (BQ, C+2, n)
        disp = self.fold_mlp(x).transpose(1, 2).reshape(B, Q, n, 3)  # (B, Q, n, 3)

        fine = (coarse.unsqueeze(2).expand(B, Q, n, 3) + disp).reshape(B, Q * n, 3)
        return coarse, fine


# ── Main model ────────────────────────────────────────────────────────────────

class PoinTr(nn.Module):
    """
    PoinTr for MM-Fi.

    Args:
        num_centers    : proxy points sampled from input (default 128)
        k              : KNN neighbours per proxy point  (default 8)
        num_queries    : decoder query tokens            (default 128)
        num_per_token  : fine points per query (perfect square, default 16 → 2048 total)
        feat_dim       : transformer feature dim         (default 384)
        num_heads      : attention heads                 (default 6)
        num_enc_layers : transformer encoder depth       (default 6)
        num_dec_layers : transformer decoder depth       (default 6)
    """

    def __init__(self,
                 num_centers:    int = 128,
                 k:              int = 8,
                 num_queries:    int = 128,
                 num_per_token:  int = 16,
                 feat_dim:       int = 384,
                 num_heads:      int = 6,
                 num_enc_layers: int = 6,
                 num_dec_layers: int = 6):
        super().__init__()
        assert num_queries * num_per_token > 0

        self.tokenizer   = PointTokenizer(num_centers, k, feat_dim)
        self.query_embed = nn.Embedding(num_queries, feat_dim)

        enc_layer = nn.TransformerEncoderLayer(
            feat_dim, num_heads, dim_feedforward=feat_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            feat_dim, num_heads, dim_feedforward=feat_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers)

        self.generator = FoldingGenerator(feat_dim, num_per_token)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, 3) sparse mmWave point cloud
        Returns:
            coarse: (B, num_queries,         3)
            fine  : (B, num_queries*num_per_token, 3)
        """
        B = x.shape[0]

        # 1. Tokenise input into proxy tokens
        _, tokens = self.tokenizer(x)                                   # (B, Nc, D)

        # 2. Encode
        memory = self.encoder(tokens)                                   # (B, Nc, D)

        # 3. Decode: learned queries attend to encoder memory
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        output  = self.decoder(queries, memory)                         # (B, Q, D)

        # 4. Generate point clouds
        coarse, fine = self.generator(output)
        return coarse, fine
