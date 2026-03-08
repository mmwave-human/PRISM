"""
train_SkelKD.py
使用 MMFiViPC.py 数据集，返回 (skeleton, lidar, mmwave, label)
SkelKD 只需要 mmwave → skeleton，lidar 和 label 直接忽略
"""

import argparse
import os
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

from datasets.MMFiViPC import get_data_loaders
from tools.io import dict2namespace
from tools.utils import AverageMeter, common_init
from completion_trainer.SkelKD_Trainer import SkelKDTrainer


def main(args, cfg):
    common_init(cfg.common.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loaders      = get_data_loaders(cfg.data)
    train_loader = loaders['train_loader']
    test_loader  = loaders['test_loader']

    trainer = SkelKDTrainer(cfg, device)

    if args.resume:
        trainer.resume(epoch=args.resume_epoch,
                       strict=args.strict,
                       load_optim=args.load_optimizer)
        trainer.optimizer.defaults['lr'] = cfg.opt.lr

    if args.evaluate:
        res = trainer.evaluate(test_loader)
        trainer.info(f"Eval only: {res}")
        return

    best_mpjpe  = float('inf')
    total_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    bone_meter  = AverageMeter()

    for epoch in range(trainer.epoch, cfg.common.epochs + 1):

        if trainer.itr > cfg.opt.warmup_iters:
            trainer.scheduler.step()

        tbar = tqdm(train_loader, ncols=160)
        tbar.set_description(f'Epoch {epoch}')

        for data in tbar:
            # MMFiViPC 返回 (skeleton, lidar, mmwave, label)
            skeleton, _, mmwave, _ = data
            mmwave   = mmwave.float().to(device)    # (B, M, 3)
            skeleton = skeleton.float().to(device)  # (B, 17, 3)

            total, mpjpe, bone = trainer.update(mmwave, skeleton)

            total_meter.update(total.detach())
            mpjpe_meter.update(mpjpe.detach())
            bone_meter.update(bone.detach())

            tbar.set_postfix({
                'loss':  f'{total_meter.val:.5f}({total_meter.avg:.5f})',
                'mpjpe': f'{mpjpe_meter.val:.5f}({mpjpe_meter.avg:.5f})',
                'bone':  f'{bone_meter.val:.5f}({bone_meter.avg:.5f})',
            })

        if epoch % cfg.log.log_epoch_freq == 0:
            trainer.updata_time()
            trainer.write_log(
                [epoch, trainer.itr,
                 total_meter.avg.item(),
                 mpjpe_meter.avg.item(),
                 bone_meter.avg.item(),
                 trainer.time],
                mode='train'
            )

        trainer.epoch_end()

        if (trainer.epoch - 1) % cfg.log.eval_epoch_freq == 0:
            res = trainer.evaluate(test_loader)
            trainer.info(f"epoch {trainer.epoch - 1}: {res}")
            try:
                trainer.write_log(
                    [trainer.epoch - 1,
                     res['mpjpe_mm'],
                     res['pa_mpjpe_mm']],
                    mode='eval'
                )
            except Exception as e:
                print(f'write eval log failed: {e}')

            if res['mpjpe_mm'] < best_mpjpe:
                best_mpjpe = res['mpjpe_mm']
                trainer.save_best(best_mpjpe)

        total_meter.reset()
        mpjpe_meter.reset()
        bone_meter.reset()

    trainer.info(f'Training finished. Best MPJPE = {best_mpjpe:.2f} mm')


def get_parser():
    parser = argparse.ArgumentParser('Train SkelKD')
    parser.add_argument('--save',           default='experiments', type=str)
    parser.add_argument('--resume',         default=False, type=eval,
                        choices=[True, False])
    parser.add_argument('--resume_epoch',   default=None,  type=int)
    parser.add_argument('--run_dir',        default=None,  type=str,
                        help='resume 时指定已有训练目录，例如: '
                             'experiments/SkelKD/mmfi_skelkd_v3_202603061200')
    parser.add_argument('--evaluate',       default=False, type=eval,
                        choices=[True, False])
    parser.add_argument('--strict',         default=True,  type=eval,
                        choices=[True, False])
    parser.add_argument('--load_optimizer', default=True,  type=eval,
                        choices=[True, False])
    return parser.parse_args()


def get_config(args):
    # config 固定放在 experiments/SkelKD/config.yaml
    # 运行结果放在带时间戳的子目录下
    path = os.path.join(args.save, 'SkelKD', 'config.yaml')
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.resume and args.run_dir:
        # resume 时使用指定目录
        cfg['log']['save_path'] = args.run_dir
    else:
        # 新训练：自动生成时间戳目录
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        run_name  = f"mmfi_skelkd_{timestamp}"
        cfg['log']['save_path'] = os.path.join(
            args.save, 'SkelKD', run_name)

    return dict2namespace(cfg)


if __name__ == '__main__':
    args = get_parser()
    cfg  = get_config(args)
    main(args, cfg)