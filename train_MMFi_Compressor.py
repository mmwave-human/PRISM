"""
train_MMFi_Compressor.py
========================
训练 Compressor（VAE 点云编解码器）on MMFi 数据集。

★ 修改记录（class_condition 支持）：
  - 训练循环：取出 label 并传入 trainer.model(pc, label=label)
  - _reconstrustion：取出 label 并传入 trainer.model(pc, label=label)
  - 需配合 config.yaml 中 model.class_condition: true 使用

目录结构：
  experiments/
  └── Compressor_Trainer/
      ├── config.yaml                        ← 配置文件固定位置
      ├── mmfi_compressor_202603021010/      ← 每次训练自动生成时间戳目录
      │   ├── checkpt_100.pth
      │   ├── training.csv
      │   └── eval.csv
      └── mmfi_compressor_202603051430/      ← 另一次训练

resume 用法：
  python train_MMFi_Compressor.py --resume True \
      --run_dir experiments/Compressor_Trainer/mmfi_compressor_202603021010
"""

import argparse
import os
import torch
import yaml
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm

from datasets.MMFiViPC import get_data_loaders
from model.Compressor.Network import Compressor
from model.Compressor.layers import index_points
from evaluation.loss import EMD_loss, CD_loss
from tools.io import dict2namespace
from tools.utils import AverageMeter, common_init
from completion_trainer.Compressor_Trainer import Trainer   # ★ 正确路径


def main(args, cfg):
    common_init(cfg.common.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loaders      = get_data_loaders(cfg.data)
    train_loader = loaders['train_loader']
    test_loader  = loaders['test_loader']

    # Compressor 训练只需要 Compressor 本身，不需要 Score 模型
    model   = Compressor(cfg.model)
    trainer = Trainer(cfg, model, device)

    if args.resume:
        trainer.resume(epoch=args.resume_epoch, finetune=args.finetune,
                       strict=args.strict, load_optim=args.load_optimizer)
        trainer.optimizer.defaults['lr'] = cfg.opt.lr

    loss_meter    = AverageMeter()
    kl_loss_meter = AverageMeter()
    rec_meter     = AverageMeter()
    max_meter     = AverageMeter()

    if not args.evaluate:
        for epoch in range(trainer.epoch, cfg.common.epochs + 1):
            if trainer.itr > cfg.opt.warmup_iters:
                trainer.scheduler.step(trainer.epoch)

            tbar = tqdm(train_loader, ncols=160)
            tbar.set_description("Epoch {}".format(epoch))

            for data in tbar:
                # ★ 取出 label（第 4 个元素），传给 Compressor
                _, pc, _, label = data
                pc    = pc.to('cuda')
                label = label.to('cuda')      # ★ label 移到 GPU

                pc_center = pointnet2_utils.furthest_point_sample(
                    pc, cfg.data.tr_max_sample_points).long()
                pc = index_points(pc, pc_center)

                trainer.model.train()
                trainer.warm_up(trainer.optimizer, trainer.itr)

                # ★ 传入 label（class_condition=false 时模型内部会忽略它，兼容）
                output      = trainer.model(pc, label=label)
                output_set  = output['set']
                kls         = output['kls']
                max_feature = output['max']

                cd_loss  = CD_loss(output_set, pc)
                rec_loss = cd_loss
                kl_loss  = torch.cat(kls, dim=1).mean()
                loss     = cfg.opt.kl_weight * kl_loss + rec_loss

                trainer.optimizer.zero_grad()
                loss.backward()
                if cfg.opt.grad_norm_clip_value is not None:
                    clip_grad_norm_(trainer.model.parameters(),
                                    cfg.opt.grad_norm_clip_value)
                trainer.optimizer.step()
                trainer.itr += 1

                loss_meter.update(loss.detach())
                kl_loss_meter.update(kl_loss.detach())
                rec_meter.update(rec_loss.detach())
                max_meter.update(max_feature.detach())

                tbar.set_postfix({
                    'loss': '{0:1.5f}({1:1.5f})'.format(
                        loss_meter.val, loss_meter.avg),
                    'kl':  '{0:1.5f}({1:1.5f})'.format(
                        kl_loss_meter.val, kl_loss_meter.avg),
                    'rec': '{0:1.5f}({1:1.5f})'.format(
                        rec_meter.val, rec_meter.avg),
                    'max': '{0:1.5f}({1:1.5f})'.format(
                        max_meter.val, max_meter.avg),
                })

                if torch.isnan(loss_meter.avg) or torch.isinf(loss_meter.avg):
                    break

            if trainer.epoch % cfg.log.log_epoch_freq == 0:
                trainer.updata_time()
                trainer.write_log(
                    [epoch, trainer.itr, loss_meter.avg,
                     kl_loss_meter.avg, rec_meter.avg,
                     max_meter.avg, trainer.time],
                    mode="train")

            trainer.epoch_end()

            if (trainer.epoch - 1) % cfg.log.eval_epoch_freq == 0:
                all_res = _reconstrustion(trainer, test_loader, cfg)
                trainer.info(f"epoch{trainer.epoch-1}: {all_res}")
                try:
                    trainer.write_log(
                        [trainer.epoch - 1] + list(all_res.values()),
                        mode="eval")
                except Exception:
                    print("write log failed")

            # NaN 保护：回退到最近一个已存在的 checkpoint
            if torch.isnan(loss_meter.avg) or torch.isinf(loss_meter.avg):
                freq = cfg.log.save_epoch_freq
                # 从当前 epoch 向前找最近的已存在 checkpoint
                back_epoch = (trainer.epoch - 1) // freq * freq
                while back_epoch > 0:
                    ckpt_path = os.path.join(
                        cfg.log.save_path, f'checkpt_{back_epoch}.pth')
                    if os.path.exists(ckpt_path):
                        print(f'  [NaN] 回退到 epoch {back_epoch}，LR 减半')
                        trainer.resume(epoch=back_epoch, finetune=False,
                                       strict=True, load_optim=True)
                        trainer.optimizer.defaults['lr'] = cfg.opt.lr / 2
                        loss_meter.reset(); kl_loss_meter.reset()
                        rec_meter.reset(); max_meter.reset()
                        break
                    back_epoch -= freq
                else:
                    print('  [NaN] 无可用 checkpoint，终止训练')
                    break

            loss_meter.reset()
            kl_loss_meter.reset()
            rec_meter.reset()
            max_meter.reset()

    else:
        all_res = _reconstrustion(trainer, test_loader, cfg)
        trainer.write_log(
            [trainer.epoch - 1] + list(all_res.values()),
            mode="eval")


def _reconstrustion(trainer, test_loader, cfg):
    """在测试集上跑重建，保存 npy 并计算分布级指标。"""
    import numpy as np
    from evaluation import compute_all_metrics

    with torch.no_grad():
        trainer.model.eval()
        all_ref, all_rec = [], []
        tbar = tqdm(test_loader)

        # ★ 取出 label 并传给模型
        for _, pc, _, label in tbar:
            pc    = pc.to('cuda')
            label = label.to('cuda')          # ★

            pc_center = pointnet2_utils.furthest_point_sample(
                pc, cfg.data.tr_max_sample_points).long()
            pc = index_points(pc, pc_center)

            output  = trainer.model(pc, label=label)   # ★
            rec_pts = output["set"]
            all_rec.append(rec_pts)
            all_ref.append(pc)

        rec = torch.cat(all_rec, dim=0)
        ref = torch.cat(all_ref, dim=0)

        np.save(
            os.path.join(cfg.log.save_path,
                         'rec_ep%d.npy' % trainer.epoch),
            rec.detach().cpu().numpy()
        )
        gen_res = compute_all_metrics(rec, ref, batch_size=64)

    all_res = {
        ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
        for k, v in gen_res.items()
    }
    print(f"Reconstruction Epoch:{trainer.epoch}", gen_res)
    return all_res


def get_parser():
    parser = argparse.ArgumentParser('MMFi Compressor')
    parser.add_argument("--dataset",        default='mmfi_compressor', type=str)
    parser.add_argument('--trainer_type',   default='Compressor_Trainer', type=str)
    parser.add_argument('--gpu',            default=0,     type=int)
    parser.add_argument('--save',           default='experiments', type=str)
    parser.add_argument('--resume',         default=False, type=eval, choices=[True, False])
    parser.add_argument('--resume_epoch',   default=None,  type=int)
    parser.add_argument('--run_dir',        default=None,  type=str,
                        help='resume 时指定已有训练目录，例如: '
                             'experiments/Compressor_Trainer/mmfi_compressor_202603021010')
    parser.add_argument('--evaluate',       default=False, type=eval, choices=[True, False])
    parser.add_argument('--strict',         default=True,  type=eval, choices=[True, False])
    parser.add_argument('--finetune',       default=False, type=eval, choices=[True, False])
    parser.add_argument('--load_optimizer', default=True,  type=eval, choices=[True, False])
    return parser.parse_args()


def get_config(args):
    path = os.path.join(args.save, args.trainer_type, "config.yaml")
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.resume and args.run_dir:
        cfg['log']['save_path'] = args.run_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        run_name = f"{args.dataset}_{timestamp}"
        cfg['log']['save_path'] = os.path.join(
            args.save, args.trainer_type, run_name)

    return dict2namespace(cfg)


if __name__ == "__main__":
    args = get_parser()
    cfg  = get_config(args)
    main(args, cfg)