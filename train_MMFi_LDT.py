"""
train_MMFi_LDT.py
=================
训练 SKD-Net 条件扩散模型。

支持两种条件模式（由 config.score.coarse_condition 控制）：
  use_coarse=True  (默认):
      mmWave pc_part → AutoEncoderE → coarse_pts (B,128,3)
      condition = {'skeleton': skeleton, 'coarse': coarse_pts}
      辅助 loss: CD(coarse_pts, LiDAR_GT) × coarse_loss_weight

  use_coarse=False (诊断/消融):
      condition = {'skeleton': skeleton}
      AutoEncoderE 不参与前向，无辅助 loss
      → 用于验证骨骼条件路径是否有效
"""

import argparse
import os
import torch
import yaml
from datetime import datetime
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm

from datasets.MMFiViPC import get_data_loaders
from model.Compressor.Network import Compressor
from model.Compressor.layers import index_points
from model.AutoEncoderE.Network import AutoEncoderE
from tools.io import dict2namespace
from tools.utils import AverageMeter, common_init
from completion_trainer.Latent_SDE_Trainer import Trainer
from model.scorenet.score import Score


def cd_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    双向 Chamfer Distance（L2），用于 coarse_pts 辅助监督。
    pred, gt: (B, N, 3)
    """
    B, N, _ = pred.shape
    _, M, _ = gt.shape
    dist = torch.sum(
        (pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2, dim=-1)
    d_pred_gt = dist.min(dim=2).values.mean()
    d_gt_pred = dist.min(dim=1).values.mean()
    return d_pred_gt + d_gt_pred


def main(args, cfg):
    common_init(cfg.common.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── 是否启用粗糙点云条件 ────────────────────────────────
    use_coarse = getattr(cfg.score, 'coarse_condition', True)
    print(f"[Config] use_coarse={use_coarse}")

    # ── 模型实例化 ──────────────────────────────────────────
    model      = Score(cfg.score)
    compressor = Compressor(cfg.compressor)

    ae_cfg = getattr(cfg, 'autoencoder_e', None)
    if use_coarse:
        autoencoder_e = AutoEncoderE(
            cfg        = ae_cfg,
            feat_dim   = getattr(ae_cfg, 'feat_dim',   256) if ae_cfg else 256,
            num_coarse = getattr(ae_cfg, 'num_coarse', 128) if ae_cfg else 128,
        ).to(device)
        coarse_loss_weight = getattr(ae_cfg, 'coarse_loss_weight', 0.1) \
                             if ae_cfg else 0.1
    else:
        autoencoder_e      = None
        coarse_loss_weight = 0.0

    loaders      = get_data_loaders(cfg.data)
    train_loader = loaders['train_loader']
    test_loader  = loaders['test_loader']

    # ── Trainer（管理 Score + EMA optimizer）───────────────
    trainer = Trainer(cfg, model=model, compressor=compressor, device=device)

    # AutoEncoderE 参数仅在 use_coarse 时加入 optimizer
    if use_coarse and autoencoder_e is not None:
        trainer.optimizer.add_param_group({
            'params': autoencoder_e.parameters(),
            'lr':     cfg.opt.lr,
        })

    # ── Resume / 加载预训练 ─────────────────────────────────
    if args.resume:
        trainer.resume(epoch=args.resume_epoch, strict=args.strict,
                       load_optim=args.load_optimizer, finetune=args.finetune)
        if use_coarse and autoencoder_e is not None:
            ae_ckpt = os.path.join(
                cfg.log.save_path,
                f'autoencoder_e_{args.resume_epoch}.pth')
            if os.path.exists(ae_ckpt):
                autoencoder_e.load_state_dict(
                    torch.load(ae_ckpt, map_location=device))
                print(f"[AutoEncoderE] loaded from {ae_ckpt}")
            else:
                print(f"[AutoEncoderE] no checkpoint at {ae_ckpt}, "
                      "using random init")
        trainer.optimizer.defaults['lr'] = cfg.opt.lr
        trainer.itr = 0
    else:
        trainer.load_pretrain()

    loss_meter      = AverageMeter()
    coarse_cd_meter = AverageMeter()

    if not args.evaluate:
        for epoch in range(trainer.epoch, cfg.common.epochs + 1):
            if trainer.itr > cfg.opt.warmup_iters:
                trainer.scheduler.step(trainer.epoch)

            tbar = tqdm(train_loader, ncols=180)
            tbar.set_description("Epoch {}".format(epoch))

            for data in tbar:
                skeleton, pc, pc_part, label = data
                pc       = pc.to(device)
                pc_part  = pc_part.to(device)
                skeleton = skeleton.float().to(device)   # (B, 17, 3)
                label    = label.to(device)              # (B,)

                # LiDAR GT FPS 至 256 点
                pc_center = pointnet2_utils.furthest_point_sample(
                    pc, cfg.data.tr_max_sample_points).long()
                pc = index_points(pc, pc_center)         # (B, 256, 3)

                # ── 构建 condition ─────────────────────────
                if use_coarse:
                    autoencoder_e.train()
                    coarse_pts, _ = autoencoder_e(pc_part)   # (B, 128, 3)
                    condition = {
                        'skeleton': skeleton,
                        'coarse':   coarse_pts.detach(),
                    }
                else:
                    coarse_pts = None
                    condition  = {'skeleton': skeleton}

                # ── 扩散 loss（内部 backward + step）────────
                diff_loss = trainer.update(pc, condition, label=label)
                loss_meter.update(diff_loss.detach())

                # ── 粗糙点云辅助 CD loss（仅 use_coarse）────
                if use_coarse:
                    coarse_cd = cd_loss(coarse_pts, pc)
                    coarse_cd_meter.update(coarse_cd.detach())
                    trainer.optimizer.zero_grad()
                    (coarse_loss_weight * coarse_cd).backward()
                    trainer.optimizer.step()

                # ── 进度条 ────────────────────────────────
                postfix = {'diff': f'{loss_meter.val:.5f}({loss_meter.avg:.5f})'}
                if use_coarse:
                    postfix['coarse'] = (f'{coarse_cd_meter.val:.5f}'
                                         f'({coarse_cd_meter.avg:.5f})')
                tbar.set_postfix(postfix)

            trainer.epoch_end()

            if (trainer.epoch - 1) % cfg.log.log_epoch_freq == 0:
                trainer.updata_time()
                log_vals = [epoch, trainer.itr,
                            loss_meter.avg,
                            coarse_cd_meter.avg if use_coarse else 0.0,
                            trainer.time]
                trainer.write_log(log_vals, "train")
                loss_meter.reset()
                coarse_cd_meter.reset()

            if (trainer.epoch - 1) % cfg.log.eval_epoch_freq == 0:
                all_res = trainer.valsample(
                    test_loader=test_loader,
                    autoencoder_e=autoencoder_e if use_coarse else None)
                trainer.info(f"epoch{trainer.epoch-1}: {all_res}")
                try:
                    trainer.write_log(
                        [trainer.epoch - 1] + list(all_res.values()), "eval")
                except Exception:
                    pass

                # 保存 AutoEncoderE checkpoint（仅 use_coarse）
                if use_coarse and autoencoder_e is not None:
                    ae_save_path = os.path.join(
                        cfg.log.save_path,
                        f'autoencoder_e_{trainer.epoch - 1}.pth')
                    torch.save(autoencoder_e.state_dict(), ae_save_path)
                    print(f"[AutoEncoderE] saved to {ae_save_path}")

    else:
        all_res = trainer.valsample(
            test_loader=test_loader,
            vis=True,
            autoencoder_e=autoencoder_e if use_coarse else None)
        trainer.write_log(
            [trainer.epoch - 1] + list(all_res.values()), "eval")


# ─────────────────────────────────────────────────────────────
#  参数解析 & 配置加载
# ─────────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser('MMFi LDT')
    parser.add_argument("--dataset",        default='mmfi',       type=str)
    parser.add_argument('--trainer_type',   default="Latent_Diffusion_Trainer", type=str)
    parser.add_argument('--gpu',            default=0,            type=int)
    parser.add_argument('--save',           default='experiments', type=str)
    parser.add_argument('--resume',         default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--resume_epoch',   default=None,         type=int)
    parser.add_argument('--run_dir',        default=None,         type=str,
                        help='resume 时指定已有训练目录（覆盖 config 里的 save_path）')
    parser.add_argument('--evaluate',       default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--strict',         default=True,
                        type=eval, choices=[True, False])
    parser.add_argument('--finetune',       default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--load_optimizer', default=False,
                        type=eval, choices=[True, False])
    return parser.parse_args()


def get_config(args):
    path = os.path.join(args.save, args.trainer_type, "config.yaml")
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.resume and args.run_dir:
        cfg['log']['save_path'] = args.run_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        run_name  = f"{args.dataset}_{timestamp}"
        cfg['log']['save_path'] = os.path.join(
            args.save, args.trainer_type, run_name)

    return dict2namespace(cfg)


if __name__ == "__main__":
    args = get_parser()
    cfg  = get_config(args)
    main(args, cfg)