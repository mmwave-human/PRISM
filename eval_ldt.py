"""
eval_ldt.py
===========
独立评估脚本：加载 LDT checkpoint，在 test set 上做条件生成，
逐样本归一化后计算 CD-based 指标（跳过 EMD，避免 OOM）。

用法：
  python eval_ldt.py \
      --run_dir experiments/Latent_Diffusion_Trainer/mmfi_202603090917 \
      --resume_epoch 400

  # 如果只想用已保存的 npy（不重新推理）：
  python eval_ldt.py \
      --run_dir experiments/Latent_Diffusion_Trainer/mmfi_202603090917 \
      --resume_epoch 400 \
      --from_npy
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pointnet2_ops import pointnet2_utils

from datasets.MMFiViPC import get_data_loaders
from model.Compressor.Network import Compressor
from model.Compressor.layers import index_points
from model.GeoAE.Network import GeoAE
from model.scorenet.score import Score
from tools.io import dict2namespace
from tools.utils import common_init
from completion_trainer.Latent_SDE_Trainer import Trainer
from evaluation.evaluation_metrics import compute_CD_metrics, compute_paired_CD


# ── 逐样本归一化 ─────────────────────────────────────────────
def normalize_pc(pc):
    """(B, N, 3) → 每个样本独立零均值 + 单位最大半径"""
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    furthest = torch.amax(
        torch.sqrt(torch.sum(pc ** 2, dim=-1, keepdim=True)),
        dim=1, keepdim=True)
    return pc / (furthest + 1e-8)


def main():
    parser = argparse.ArgumentParser('LDT Eval (CD-only)')
    parser.add_argument('--run_dir',      required=True, type=str)
    parser.add_argument('--resume_epoch', required=True, type=int)
    parser.add_argument('--from_npy',     action='store_true',
                        help='跳过推理，直接从已保存的 smp/ref npy 计算指标')
    parser.add_argument('--max_samples',  type=int, default=0,
                        help='限制最大样本数（0=全部）减少显存占用')
    parser.add_argument('--gpu',          type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    epoch  = args.resume_epoch

    # ── 加载 config ──────────────────────────────────────────
    cfg_path = os.path.join(
        os.path.dirname(args.run_dir), 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader))
    cfg.log.save_path = args.run_dir

    if args.from_npy:
        # ════════════════════════════════════════════════════════
        #  模式 A：直接从 npy 加载（不需要模型推理）
        # ════════════════════════════════════════════════════════
        smp_path = os.path.join(args.run_dir, f'smp_ep{epoch + 1}.npy')
        ref_path = os.path.join(args.run_dir, f'ref_ep{epoch + 1}.npy')
        lbl_path = os.path.join(args.run_dir, f'lbl_ep{epoch + 1}.npy')

        if not os.path.exists(smp_path):
            smp_path = os.path.join(args.run_dir, f'smp_ep{epoch}.npy')
            ref_path = os.path.join(args.run_dir, f'ref_ep{epoch}.npy')
            lbl_path = os.path.join(args.run_dir, f'lbl_ep{epoch}.npy')

        print(f'Loading smp: {smp_path}')
        print(f'Loading ref: {ref_path}')
        smp    = torch.from_numpy(np.load(smp_path, allow_pickle=True)).to(device)
        ref    = torch.from_numpy(np.load(ref_path, allow_pickle=True)).to(device)
        labels = torch.from_numpy(np.load(lbl_path, allow_pickle=True)) \
                 if os.path.exists(lbl_path) else None

    else:
        # ════════════════════════════════════════════════════════
        #  模式 B：加载模型，重新推理
        # ════════════════════════════════════════════════════════
        common_init(cfg.common.seed)

        use_coarse = getattr(cfg.score, 'coarse_condition', True)

        model      = Score(cfg.score)
        compressor = Compressor(cfg.compressor)
        trainer    = Trainer(cfg, model=model, compressor=compressor, device=device)

        ae_cfg = getattr(cfg, 'geo_ae', None)
        if use_coarse:
            geo_ae = GeoAE(
                cfg        = ae_cfg,
                feat_dim   = getattr(ae_cfg, 'feat_dim',   256) if ae_cfg else 256,
                num_coarse = getattr(ae_cfg, 'num_coarse', 128) if ae_cfg else 128,
            ).to(device)
        else:
            geo_ae = None

        # 加载权重
        trainer.resume(epoch=epoch, strict=True, load_optim=False)

        if use_coarse and geo_ae is not None:
            ae_ckpt = os.path.join(args.run_dir, f'geo_ae_{epoch}.pth')
            if os.path.exists(ae_ckpt):
                geo_ae.load_state_dict(
                    torch.load(ae_ckpt, map_location=device))
                print(f'[GeoAE] loaded from {ae_ckpt}')
            else:
                print(f'[GeoAE] WARNING: {ae_ckpt} 不存在，使用随机初始化')
            geo_ae.eval()

        loaders     = get_data_loaders(cfg.data)
        test_loader = loaders['test_loader']

        # ── 推理 ──────────────────────────────────────────────
        all_ref, all_smp, all_labels = [], [], []
        use_time = 0.

        with torch.no_grad():
            trainer.model.eval()
            trainer.compressor.eval()
            tbar = tqdm(test_loader, desc='Generating')

            for data in tbar:
                skeleton, pc, pc_part, label = data
                pc       = pc.to(device)
                pc_part  = pc_part.to(device)
                skeleton = skeleton.float().to(device)
                label    = label.to(device)

                pc_center = pointnet2_utils.furthest_point_sample(pc, 256).long()
                ref_pts   = index_points(pc, pc_center)

                pc_part_center = pointnet2_utils.furthest_point_sample(
                    pc_part, pc_part.size(1)).long()
                pc_part = index_points(pc_part, pc_part_center)

                if use_coarse and geo_ae is not None:
                    coarse_pts, _ = geo_ae(pc_part)
                    condition = {'skeleton': skeleton, 'coarse': coarse_pts}
                else:
                    condition = {'skeleton': skeleton}

                T = time.time()
                smp_pts, _ = trainer.sample(
                    num_samples=ref_pts.size(0),
                    label=label,
                    condition=condition)
                use_time += time.time() - T

                all_smp.append(smp_pts)
                all_ref.append(ref_pts)
                all_labels.append(label.cpu())

        smp    = torch.cat(all_smp,    dim=0)
        ref    = torch.cat(all_ref,    dim=0)
        labels = torch.cat(all_labels, dim=0)
        print(f'Inference done: {smp.shape[0]} samples, '
              f'rate = {smp.shape[0] / use_time:.2f} samples/s')

        # 保存 npy
        np.save(os.path.join(args.run_dir, f'smp_ep{epoch + 1}.npy'),
                smp.cpu().numpy())
        np.save(os.path.join(args.run_dir, f'ref_ep{epoch + 1}.npy'),
                ref.cpu().numpy())
        np.save(os.path.join(args.run_dir, f'lbl_ep{epoch + 1}.npy'),
                labels.numpy())

    # ════════════════════════════════════════════════════════════
    #  计算指标（CD-only，跳过 EMD 避免 OOM）
    # ════════════════════════════════════════════════════════════
    N = smp.shape[0]
    print(f'\n总样本数: {N}')

    if args.max_samples > 0 and N > args.max_samples:
        idx = torch.randperm(N)[:args.max_samples]
        smp = smp[idx]
        ref = ref[idx]
        print(f'下采样到 {args.max_samples} 个样本')

    # ── 诊断打印 ─────────────────────────────────────────────
    print(f'\n[DIAG] BEFORE normalize:')
    print(f'  ref  range: [{ref.min():.4f}, {ref.max():.4f}]  '
          f'mean_norm: {torch.sqrt((ref**2).sum(-1)).mean():.4f}')
    print(f'  smp  range: [{smp.min():.4f}, {smp.max():.4f}]  '
          f'mean_norm: {torch.sqrt((smp**2).sum(-1)).mean():.4f}')

    # ── 逐样本归一化 ─────────────────────────────────────────
    smp = normalize_pc(smp)
    ref = normalize_pc(ref)

    print(f'[DIAG] AFTER normalize:')
    print(f'  ref  range: [{ref.min():.4f}, {ref.max():.4f}]  '
          f'mean_norm: {torch.sqrt((ref**2).sum(-1)).mean():.4f}')
    print(f'  smp  range: [{smp.min():.4f}, {smp.max():.4f}]  '
          f'mean_norm: {torch.sqrt((smp**2).sum(-1)).mean():.4f}')

    # ── 分布级 CD 指标（1-NNA / MMD / COV）──────────────────
    print(f'\n计算分布级 CD 指标（跳过 EMD）...')
    gen_res = compute_CD_metrics(smp, ref, batch_size=64)

    print('\n' + '=' * 60)
    print(f'  Epoch {epoch} 评估结果（逐样本归一化 + CD-only）')
    print('=' * 60)
    for k, v in gen_res.items():
        val = v if isinstance(v, float) else v.item()
        print(f'  {k:20s} = {val:.6f}')
    print('=' * 60)

    cov_cd = gen_res.get('cov-CD', None)
    nn_cd  = gen_res.get('1-NN-CD-acc', None)
    mmd_cd = gen_res.get('mmd-CD', None)
    if cov_cd  is not None:
        print(f'\n  ★ COV-CD  = {(cov_cd  if isinstance(cov_cd,  float) else cov_cd.item()) *100:.2f}%')
    if nn_cd   is not None:
        print(f'  ★ 1-NN-CD = {(nn_cd   if isinstance(nn_cd,   float) else nn_cd.item())  *100:.2f}%')
    if mmd_cd  is not None:
        print(f'  ★ MMD-CD  = {(mmd_cd  if isinstance(mmd_cd,  float) else mmd_cd.item())      :.6f}')

    # ── 配对 CD（条件生成的核心指标）────────────────────────
    print(f'\n计算配对 rec-CD（smp[i] vs ref[i]，逐样本配对）...')
    _ACTION_NAMES = {
        0: 'A03', 1: 'A12', 2: 'A13', 3: 'A17',
        4: 'A19', 5: 'A22', 6: 'A26', 7: 'A27',
    }
    cd_per_sample, rec_cd_mean = compute_paired_CD(smp, ref, batch_size=64)
    print('\n' + '=' * 60)
    print(f'  ★ rec-CD (全部均值) = {rec_cd_mean:.6f}')
    if labels is not None:
        print('  -- 分动作 rec-CD --')
        for cls_idx in range(8):
            mask = (labels == cls_idx)
            if mask.sum() > 0:
                mean_cls = cd_per_sample[mask.to(cd_per_sample.device)].mean().item()
                name = _ACTION_NAMES.get(cls_idx, str(cls_idx))
                print(f'    {name} (n={mask.sum():3d}) : {mean_cls:.6f}')
    print('=' * 60)


if __name__ == '__main__':
    main()