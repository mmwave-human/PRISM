"""
loss_utils.py
=============
Chamfer Distance utilities for PCN training and evaluation.
Uses the compiled chamfer3D extension (installed from mmPoint).
"""

import os
import sys
import torch

# chamfer3D is built inside mmPoint; add its parent to sys.path
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_chamfer_dir = os.path.join(_here, '..', 'mmPoint', 'utils', 'ChamferDistancePytorch')
sys.path.insert(0, os.path.abspath(_chamfer_dir))
from chamfer3D import dist_chamfer_3D

_chamfer = dist_chamfer_3D.chamfer_3DDist()


def calc_cd(output, gt):
    """
    Compute CD-L1 and CD-L2 between output and gt.
    Matches mmPoint's calc_cd convention for fair comparison.

    Args:
        output: (B, N, 3)
        gt    : (B, M, 3)
    Returns:
        cd_l1: (B,)  mean L1 chamfer distance  (×1e4 in logging)
        cd_l2: (B,)  sum  L2 chamfer distance  (×1e4 in logging)
    """
    dist1, dist2, _, _ = _chamfer(gt.contiguous(), output.contiguous())
    cd_l1 = (dist1.sqrt().mean(1) + dist2.sqrt().mean(1)) / 2.0
    cd_l2 = dist1.mean(1) + dist2.mean(1)
    return cd_l1, cd_l2


def pcn_loss(coarse, fine, gt, alpha: float = 0.5):
    """
    PCN training loss:
      L = CD_L1(coarse, gt_coarse) + alpha × CD_L1(fine, gt)

    gt_coarse is a random subsample of gt to num_coarse points.

    Returns:
        total_loss, loss_coarse, loss_fine  — all scalars
    """
    B, Nc, _ = coarse.shape
    idx = torch.randperm(gt.shape[1], device=gt.device)[:Nc]
    gt_coarse = gt[:, idx, :]

    d1c, d2c, _, _ = _chamfer(gt_coarse.contiguous(), coarse.contiguous())
    loss_coarse = (d1c.sqrt().mean(1) + d2c.sqrt().mean(1)).mean()

    d1f, d2f, _, _ = _chamfer(gt.contiguous(), fine.contiguous())
    loss_fine   = (d1f.sqrt().mean(1) + d2f.sqrt().mean(1)).mean()

    return loss_coarse + alpha * loss_fine, loss_coarse, loss_fine
