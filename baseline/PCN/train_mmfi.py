"""
train_mmfi.py — PCN baseline on MM-Fi
======================================
Usage:
    cd baseline/PCN
    CUDA_VISIBLE_DEVICES=0 python train_mmfi.py -c cfgs/pcn_mmfi.yaml
"""

import os
import sys
import random
import logging
import argparse
import time as timetmp
from time import time

import numpy as np
import munch
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset.dataset_mmfi import MMFiDataset
from models.pcn import PCN
from utils.loss_utils import pcn_loss, calc_cd

import warnings
warnings.filterwarnings('ignore')


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_folders(args):
    model_name = 'PCN-%s' % timetmp.strftime('%m%d_%H%M', timetmp.localtime())
    out_dir  = os.path.join(args.dir_outpath, model_name)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_fout = open(os.path.join(out_dir, 'log_%s.csv' % model_name), 'w')
    return model_name, out_dir, ckpt_dir, log_fout


def save_model(path, net):
    torch.save(net.state_dict(), path)


def build_dataset(args):
    actions    = list(args.actions)
    cache_root = getattr(args, 'cache_root', None) or None

    train_ds = MMFiDataset(
        args.data_root, list(args.train_subjects), actions,
        lidar_npoints=args.lidar_npoints, mmwave_npoints=args.mmwave_npoints,
        cache_root=cache_root)

    val_ds = MMFiDataset(
        args.data_root, list(args.val_subjects), actions,
        lidar_npoints=args.lidar_npoints, mmwave_npoints=args.mmwave_npoints,
        cache_root=cache_root)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=int(args.workers), pin_memory=True,
        prefetch_factor=4, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=int(args.workers), pin_memory=True,
        prefetch_factor=4, persistent_workers=True)

    print('Train samples: %d   Val samples: %d' % (len(train_ds), len(val_ds)))
    return train_loader, val_loader


# ── Validation ────────────────────────────────────────────────────────────────

def validate(net, epoch, val_loader, ckpt_dir, log_fout, best_l1, best_l2):
    logging.info('Validating ...')
    net.eval()
    total_l1 = total_l2 = 0
    n = len(val_loader)

    with torch.no_grad():
        for data in val_loader:
            mm = data['image'].cuda()
            gt = data['points'].cuda()
            B  = gt.shape[0]
            _, fine = net(mm)
            l1, l2 = calc_cd(fine, gt)
            total_l1 += l1.sum().item() / B * 1e4
            total_l2 += l2.sum().item() / B * 1e4

    avg_l1 = total_l1 / n
    avg_l2 = total_l2 / n

    if avg_l1 < best_l1:
        best_l1 = avg_l1
        save_model(os.path.join(ckpt_dir, 'bestl1_network.pth'), net)
        logging.info('  → new best CD-L1: %.4f  (saved)' % best_l1)

    if avg_l2 < best_l2:
        best_l2 = avg_l2
        save_model(os.path.join(ckpt_dir, 'bestl2_network.pth'), net)
        logging.info('  → new best CD-L2: %.4f  (saved)' % best_l2)

    log_fout.write('%d,%.4f,%.4f,%.4f,%.4f\n' %
                   (epoch, avg_l1, best_l1, avg_l2, best_l2))
    log_fout.flush()
    logging.info('[Epoch %d] Val CD_L1=%.4f (best %.4f)  CD_L2=%.4f (best %.4f)' %
                 (epoch, avg_l1, best_l1, avg_l2, best_l2))
    return best_l1, best_l2


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    exp_name, out_dir, ckpt_dir, log_fout = set_folders(args)
    log_fout.write('EPOCH,CD_L1,BEST_CDL1,CD_L2,BEST_CDL2\n')

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(out_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ])
    logging.info(str(args))

    # Seed
    seed = int(args.manual_seed) if args.manual_seed else random.randint(1, 10000)
    logging.info('Random seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Dataset
    train_loader, val_loader = build_dataset(args)

    # Model
    net = PCN(num_coarse=args.num_coarse, num_fine=args.num_fine,
              feat_dim=args.feat_dim, grid_size=args.grid_size).cuda()
    logging.info('PCN params: %d' % sum(p.numel() for p in net.parameters()))

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_l1 = best_l2 = float('inf')

    for epoch in range(args.start_epoch, args.nepoch):
        t0 = time()
        net.train()

        # LR decay
        if args.lr_decay and epoch > 0 and epoch % args.lr_decay_interval == 0:
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * args.lr_decay_rate, args.lr_clip)
            logging.info('LR → %.2e' % optimizer.param_groups[0]['lr'])

        total_loss = total_l1 = total_l2 = 0
        n_batches  = len(train_loader)

        with tqdm(train_loader) as t:
            for batch_idx, data in enumerate(t):
                optimizer.zero_grad()
                mm = data['image'].cuda()
                gt = data['points'].cuda()
                B  = gt.shape[0]

                coarse, fine = net(mm)
                loss, lc, lf = pcn_loss(coarse, fine, gt, alpha=args.alpha)
                loss.backward()
                optimizer.step()

                # Logging metrics (same scale as mmPoint for fair comparison)
                l1_val, l2_val = calc_cd(fine, gt)
                total_l1 += l1_val.sum().item() / B * 1e4
                total_l2 += l2_val.sum().item() / B * 1e4
                total_loss += loss.item()

                t.set_description('[E %d/%d]' % (epoch, args.nepoch))
                t.set_postfix(loss='%.4f' % loss.item(),
                              cd_l1='%.2f' % (l1_val.mean().item() * 1e4))

        logging.info('%s [Epoch %d/%d] Time=%.1fs  CD_L1=%.4f  CD_L2=%.4f' % (
            exp_name, epoch, args.nepoch, time() - t0,
            total_l1 / n_batches, total_l2 / n_batches))

        if epoch % args.epoch_interval_to_save == 0:
            save_model(os.path.join(ckpt_dir, '%d_network.pth' % epoch), net)

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_l1, best_l2 = validate(
                net, epoch, val_loader, ckpt_dir, log_fout, best_l1, best_l2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    arg  = parser.parse_args()
    args = munch.munchify(yaml.safe_load(open(arg.config)))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    train(args)
