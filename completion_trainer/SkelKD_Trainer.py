"""
trainer/SkelKD_Trainer.py

SkelKD 的独立训练器，继承 BaseTrainer 的日志/保存/warmup 机制。
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model.SkelKD.Network import SkelKD, skelkd_loss, mpjpe_loss
from tools.log import logger
from tools.io import makedirs


class SkelKDTrainer:
    """
    SkelKD 独立训练器。

    cfg 需包含：
      cfg.model          → SkelKD 模型 config
      cfg.opt.lr         → 学习率
      cfg.opt.beta1/2    → Adam betas
      cfg.opt.weight_decay
      cfg.opt.warmup_iters
      cfg.opt.grad_norm_clip_value
      cfg.opt.bone_weight       → 骨骼长度损失权重（默认 0.1）
      cfg.common.epochs
      cfg.common.seed
      cfg.log.save_path
      cfg.log.save_epoch_freq
      cfg.log.log_epoch_freq
      cfg.log.eval_epoch_freq
      cfg.log.traincolumns / trainformat
      cfg.log.evalcolumns  / evalformat
    """

    def __init__(self, cfg, device):
        self.cfg    = cfg
        self.device = device
        self.itr    = 0
        self.epoch  = 1
        self.time   = 0
        self.tmp    = time.time()

        # 模型
        self.model = SkelKD(cfg.model).to(device)

        # 优化器
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.opt.lr,
            betas=(cfg.opt.beta1, cfg.opt.beta2),
            weight_decay=cfg.opt.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, cfg.common.epochs, eta_min=1e-6
        )

        # 日志
        makedirs(cfg.log.save_path)
        self.logger = logger(cfg)

        self.bone_weight = getattr(cfg.opt, 'bone_weight', 0.1)

    # ──────────────────────────────────────────
    #  训练一步
    # ──────────────────────────────────────────
    def update(self, mmwave, skeleton_gt):
        """
        mmwave:       (B, N, 3)  稀疏毫米波点云
        skeleton_gt:  (B, 17, 3) 关节坐标 GT（深度相机 / mocap）
        return: total_loss, mpjpe, bone_loss
        """
        self.model.train()
        self._warm_up(self.itr)

        pred = self.model(mmwave)
        total, m_loss, b_loss = skelkd_loss(pred, skeleton_gt, self.bone_weight)

        self.optimizer.zero_grad()
        total.backward()
        if self.cfg.opt.grad_norm_clip_value is not None:
            clip_grad_norm_(self.model.parameters(),
                            self.cfg.opt.grad_norm_clip_value)
        self.optimizer.step()
        self.itr += 1

        return total, m_loss, b_loss

    # ──────────────────────────────────────────
    #  验证：计算 MPJPE（mm）
    # ──────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, test_loader):
        """
        返回 dict:
          mpjpe_mm  : float  平均每关节误差（毫米）
          pa_mpjpe  : float  Procrustes 对齐后误差（毫米）
        """
        self.model.eval()
        all_pred, all_gt = [], []

        for data in tqdm(test_loader, desc="Eval", ncols=100):
            # MMFiViPC 返回 (skeleton, lidar, mmwave, label)
            skeleton, _, mmwave, _ = data
            mmwave   = mmwave.float().to(self.device)
            skeleton = skeleton.float().to(self.device)

            pred = self.model(mmwave)
            all_pred.append(pred.cpu())
            all_gt.append(skeleton.cpu())

        pred_all = torch.cat(all_pred, dim=0)   # (N, 17, 3)
        gt_all   = torch.cat(all_gt,   dim=0)   # (N, 17, 3)

        # MPJPE：假设坐标单位为 米，×1000 转换为毫米
        # 若 MM-Fi 坐标已是毫米则去掉 *1000
        mpjpe = torch.norm(pred_all - gt_all, dim=-1).mean().item() * 1000

        # PA-MPJPE（Procrustes 对齐）
        pa_mpjpe = self._procrustes_mpjpe(pred_all, gt_all) * 1000

        # 保存预测结果便于后续可视化
        np.save(
            os.path.join(self.cfg.log.save_path,
                         f'pred_joints_ep{self.epoch}.npy'),
            pred_all.numpy()
        )
        np.save(
            os.path.join(self.cfg.log.save_path,
                         f'gt_joints_ep{self.epoch}.npy'),
            gt_all.numpy()
        )

        return {'mpjpe_mm': mpjpe, 'pa_mpjpe_mm': pa_mpjpe}

    # ──────────────────────────────────────────
    #  Procrustes 对齐
    # ──────────────────────────────────────────
    @staticmethod
    def _procrustes_mpjpe(pred, gt):
        """
        pred, gt: (N, J, 3)  CPU tensor
        计算 Procrustes 对齐（平移+旋转+缩放）后的 MPJPE
        return: float（与输入坐标单位相同）
        """
        errors = []
        for p, g in zip(pred, gt):
            # 去中心
            mu_p = p.mean(0, keepdim=True)
            mu_g = g.mean(0, keepdim=True)
            p_c  = p - mu_p
            g_c  = g - mu_g

            # 缩放
            scale_p = torch.norm(p_c)
            scale_g = torch.norm(g_c)
            if scale_p < 1e-8 or scale_g < 1e-8:
                errors.append(torch.norm(p - g, dim=-1).mean().item())
                continue
            p_c = p_c / scale_p
            g_c = g_c / scale_g

            # 旋转（SVD）
            H   = p_c.T @ g_c
            U, S, Vt = torch.linalg.svd(H)
            d   = torch.det(Vt.T @ U.T)
            D   = torch.diag(torch.tensor([1., 1., d], device=p.device))
            R   = Vt.T @ D @ U.T

            # 对齐
            p_aligned = scale_g * (p_c @ R.T) + mu_g
            errors.append(torch.norm(p_aligned - g, dim=-1).mean().item())

        return float(np.mean(errors))

    # ──────────────────────────────────────────
    #  Checkpoint 保存 / 加载
    # ──────────────────────────────────────────
    def save(self):
        path = os.path.join(
            self.cfg.log.save_path,
            f'checkpt_{self.epoch}.pth'
        )
        torch.save({
            'state_dict':       self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'scheduler':        self.scheduler.state_dict(),
            'epoch':            self.epoch,
            'itr':              self.itr,
            'time':             self.time,
        }, path)

    def save_best(self, mpjpe):
        path = os.path.join(self.cfg.log.save_path, 'checkpt_best.pth')
        torch.save({
            'state_dict': self.model.state_dict(),
            'epoch':      self.epoch,
            'mpjpe_mm':   mpjpe,
        }, path)
        self.logger.info(f"  [Best] saved  MPJPE={mpjpe:.2f} mm  @ epoch {self.epoch}")

    def resume(self, epoch=None, strict=True, load_optim=True):
        if epoch is None:
            path = os.path.join(self.cfg.log.save_path, 'training.csv')
            df   = pd.read_csv(path)
            epoch = int(df['epoch'].values[-1])
        ckpt_path = os.path.join(
            self.cfg.log.save_path, f'checkpt_{epoch}.pth'
        )
        ckpt = torch.load(ckpt_path,
                          map_location=lambda s, l: s)
        self.model.load_state_dict(ckpt['state_dict'], strict=strict)
        if load_optim and 'optim_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optim_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        self.epoch     = ckpt['epoch'] + 1
        self.itr       = ckpt['itr']
        self.time      = ckpt['time']
        self.scheduler.last_epoch = self.epoch
        self.logger.info(f"Resumed from epoch {epoch}")

    # ──────────────────────────────────────────
    #  工具方法
    # ──────────────────────────────────────────
    def epoch_end(self):
        if self.epoch % self.cfg.log.save_epoch_freq == 0:
            self.save()
        self.epoch += 1

    def write_log(self, message, mode='train'):
        self.logger.write(message, mode)

    def info(self, message):
        self.logger.info(message)

    def updata_time(self):
        self.time += time.time() - self.tmp
        self.tmp = time.time()

    def _warm_up(self, itr):
        warmup = self.cfg.opt.warmup_iters
        if itr < warmup:
            lr = self.cfg.opt.lr * max(float(itr + 1) / max(warmup, 1), 1e-6)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr