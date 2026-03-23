"""
pcn.py
======
Point Completion Network (PCN) adapted for MM-Fi.
Chen et al., "PCN: Point Completion Network", 3DV 2018.

Input : sparse mmWave point cloud (B, M, 3)
Output: coarse (B, num_coarse, 3)  +  fine (B, num_fine, 3)
"""

import torch
import torch.nn as nn


class PCNEncoder(nn.Module):
    """Two-level PointNet encoder from the PCN paper."""

    def __init__(self, feat_dim: int = 1024):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3,   128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(512, 512,      1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3)  →  feat: (B, feat_dim)"""
        x = x.transpose(1, 2)                              # (B, 3, N)
        f1 = self.mlp1(x)                                  # (B, 256, N)
        g1 = f1.max(dim=2)[0]                              # (B, 256)
        g1_exp = g1.unsqueeze(2).expand_as(f1)
        f2 = self.mlp2(torch.cat([f1, g1_exp], dim=1))    # (B, feat_dim, N)
        return f2.max(dim=2)[0]                            # (B, feat_dim)


class PCN(nn.Module):
    """
    PCN model for MM-Fi.

    Args:
        num_coarse : number of coarse output points  (default 512)
        num_fine   : number of fine output points    (default 2048)
        feat_dim   : global feature dimension        (default 1024)
        grid_size  : folding grid side length; must satisfy
                     num_coarse × grid_size² == num_fine  (default 2)
    """

    def __init__(self, num_coarse: int = 512, num_fine: int = 2048,
                 feat_dim: int = 1024, grid_size: int = 2):
        super().__init__()
        assert num_coarse * grid_size * grid_size == num_fine, (
            f"num_coarse ({num_coarse}) × grid_size² ({grid_size}²) "
            f"must equal num_fine ({num_fine})"
        )
        self.num_coarse = num_coarse
        self.num_fine   = num_fine
        self.grid_size  = grid_size
        self.feat_dim   = feat_dim

        self.encoder = PCNEncoder(feat_dim)

        # Coarse decoder: FC layers → num_coarse points
        self.coarse_dec = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(),
            nn.Linear(1024,     1024), nn.ReLU(),
            nn.Linear(1024,     num_coarse * 3),
        )

        # Fine decoder: folding MLP
        self.fine_dec = nn.Sequential(
            nn.Conv1d(feat_dim + 3 + 2, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512,              512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512,              3,   1),
        )

        # Fixed 2D folding grid, registered as buffer (moves to GPU with model)
        a = torch.linspace(-0.05, 0.05, steps=grid_size)
        gx, gy = torch.meshgrid(a, a)                       # (g, g), (g, g)
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (g², 2)
        self.register_buffer('grid', grid)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, 3)  sparse mmWave point cloud
        Returns:
            coarse: (B, num_coarse, 3)
            fine:   (B, num_fine,   3)
        """
        B = x.shape[0]
        gs = self.grid_size ** 2                # grid points per coarse point

        # ── Encode ────────────────────────────────────────────────────────
        feat = self.encoder(x)                  # (B, feat_dim)

        # ── Coarse output ─────────────────────────────────────────────────
        coarse = self.coarse_dec(feat).view(B, self.num_coarse, 3)  # (B, Nc, 3)

        # ── Fine output (FoldingNet) ───────────────────────────────────────
        # Repeat each coarse point grid_size² times
        c_exp = coarse.unsqueeze(2).expand(B, self.num_coarse, gs, 3)
        c_exp = c_exp.reshape(B, self.num_fine, 3)          # (B, Nf, 3)

        # Tile the 2D grid num_coarse times
        grid_exp = self.grid.unsqueeze(0).expand(self.num_coarse, gs, 2)
        grid_exp = grid_exp.reshape(self.num_fine, 2)
        grid_exp = grid_exp.unsqueeze(0).expand(B, -1, -1)  # (B, Nf, 2)

        # Repeat global feature
        feat_exp = feat.unsqueeze(1).expand(B, self.num_fine, self.feat_dim)

        # Decode displacement, add to coarse
        x_in = torch.cat([feat_exp, c_exp, grid_exp], dim=-1)  # (B, Nf, fd+5)
        disp = self.fine_dec(x_in.transpose(1, 2))             # (B, 3, Nf)
        fine = c_exp + disp.transpose(1, 2)                    # (B, Nf, 3)

        return coarse, fine
