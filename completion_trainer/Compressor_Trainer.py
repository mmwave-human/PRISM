"""
completion_trainer/Compressor_Trainer.py

★ 修改记录（class_condition 支持）：
  - compute_loss：增加 label 参数，传给 self.model(target_set, label=label)
  - reconstrustion：取出 label 并传给模型
"""

import os
import numpy as np
import pandas as pd
import torch
from pointnet2_ops import pointnet2_utils
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from model.Compressor.layers import index_points
try:
    from evaluation.loss import EMD_loss, CD_loss
except:
    pass
from evaluation import compute_all_metrics
from evaluation.evaluation_metrics import compute_CD_metrics
from trainer.base import BaseTrainer


class Trainer(BaseTrainer):

    def __init__(self, cfg, model, device):
        super(Trainer, self).__init__(cfg, device)
        self.num_points = cfg.data.tr_max_sample_points
        self.device = device
        self.kl_weight = cfg.opt.kl_weight
        self.model = model.to(device)
        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.opt.lr,
            betas=(cfg.opt.beta1, cfg.opt.beta2),
            weight_decay=cfg.opt.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg.common.epochs, 0)

    def update(self, data, label=None):
        self.model.train()
        self.warm_up(self.optimizer, self.itr)
        loss, kl_loss, rec_loss, max_feature = self.compute_loss(data, label=label)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.opt.grad_norm_clip_value is not None:
            clip_grad_norm_(self.model.parameters(),
                            self.cfg.opt.grad_norm_clip_value)
        self.optimizer.step()
        self.itr += 1
        return loss, kl_loss, rec_loss, max_feature

    def compute_loss(self, target_set, label=None):
        # ★ label 传给模型（class_condition=false 时内部忽略）
        output = self.model(target_set, label=label)
        output_set, kls, max_feature = output['set'], output['kls'], output['max']
        emd_loss = EMD_loss(output_set, target_set)
        cd_loss  = CD_loss(output_set, target_set)
        rec_loss = cd_loss + emd_loss
        kl_loss  = torch.cat(kls, dim=1).mean()
        loss     = self.kl_weight * kl_loss + rec_loss
        return loss, kl_loss, rec_loss, max_feature

    def sample(self, num_samples, num_points, given_eps=None):
        shape = (num_samples, num_points)
        self.model.eval()
        with torch.no_grad():
            sample = self.model.sample(shape, given_eps=given_eps)
        return sample

    def reconstrustion(self, test_loader):
        """验证集重建评估，传入 label。"""
        with torch.no_grad():
            self.model.eval()
            all_ref, all_rec = [], []
            tbar = tqdm(test_loader)

            # ★ 取出 label
            for _, pc, _, label in tbar:
                pc    = pc.to('cuda')
                label = label.to('cuda')      # ★

                pc_center = pointnet2_utils.furthest_point_sample(
                    pc, self.num_points).long()
                ref_pts = index_points(pc, pc_center)

                output  = self.model(ref_pts, label=label)   # ★
                rec_pts = output["set"]
                all_rec.append(rec_pts)
                all_ref.append(ref_pts)

            rec = torch.cat(all_rec, dim=0)
            ref = torch.cat(all_ref, dim=0)
            np.save(
                os.path.join(self.cfg.log.save_path,
                             'rec_ep%d.npy' % self.epoch),
                rec.detach().cpu().numpy()
            )
            gen_res = compute_CD_metrics(rec, ref, batch_size=64)
            all_res = {k: (v if isinstance(v, float) else v.item())
                       for k, v in gen_res.items()}
        print("Validation Sample (unit) Epoch:%d " % self.epoch, all_res)
        return all_res

    def resume(self, epoch=None, finetune=False, strict=False, load_optim=True):
        if epoch is None:
            path = os.path.join(self.cfg.log.save_path, 'training.csv')
            tsdf = pd.read_csv(path)
            epoch = tsdf["epoch"].values[-1]
        path = os.path.join(self.cfg.log.save_path,
                            'checkpt_{:}.pth'.format(epoch))
        checkpt = torch.load(path, map_location=lambda storage, loc: storage)
        if not finetune:
            self.model.load_state_dict(checkpt["state_dict"], strict=strict)
            if load_optim:
                if "optim_state_dict" in checkpt.keys():
                    self.optimizer.load_state_dict(checkpt["optim_state_dict"])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(self.device, non_blocking=True)
            self.epoch = checkpt["epoch"] + 1
            self.scheduler.base_lrs = [self.cfg.opt.lr]
            self.scheduler.step(self.epoch)
            self.itr  = checkpt["itr"]
            self.time = checkpt["time"]
        else:
            self.model.load_state_dict(checkpt["state_dict"], strict=False)
        self.model.init()

    def load_pretrain(self):
        path = os.path.join(self.cfg.model.pretrain_path)
        checkpt = torch.load(path)
        self.model.load_state_dict(checkpt["state_dict"], strict=True)
        self.model.init()