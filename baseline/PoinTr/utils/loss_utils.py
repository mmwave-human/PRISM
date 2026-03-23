"""
loss_utils.py — PoinTr loss and evaluation metrics.
Uses the chamfer3D extension built under mmPoint.
"""

import os
import sys
import torch

_cd_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../../mmPoint/utils/ChamferDistancePytorch'))
sys.path.insert(0, _cd_dir)
from chamfer3D import dist_chamfer_3D

_chamfer = dist_chamfer_3D.chamfer_3DDist()


def calc_cd(output, gt):
    """
    CD-L1 and CD-L2 matching mmPoint's convention (×1e4 in logging).
    Returns (cd_l1, cd_l2) each of shape (B,).
    """
    d1, d2, _, _ = _chamfer(gt.contiguous(), output.contiguous())
    cd_l1 = (d1.sqrt().mean(1) + d2.sqrt().mean(1)) / 2.0
    cd_l2 = d1.mean(1) + d2.mean(1)
    return cd_l1, cd_l2


def pointr_loss(coarse, fine, gt, alpha: float = 0.5):
    """
    L = CD_L1(coarse, gt_coarse) + alpha × CD_L1(fine, gt)
    gt_coarse: random subsample of gt to num_queries points.
    """
    B, Nc, _ = coarse.shape
    idx = torch.randperm(gt.shape[1], device=gt.device)[:Nc]
    gt_c = gt[:, idx, :]

    d1c, d2c, _, _ = _chamfer(gt_c.contiguous(), coarse.contiguous())
    loss_c = (d1c.sqrt().mean(1) + d2c.sqrt().mean(1)).mean()

    d1f, d2f, _, _ = _chamfer(gt.contiguous(), fine.contiguous())
    loss_f = (d1f.sqrt().mean(1) + d2f.sqrt().mean(1)).mean()

    return loss_c + alpha * loss_f, loss_c, loss_f
