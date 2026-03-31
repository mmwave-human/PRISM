"""
eval_baseline.py
================
对 PCN / mmPoint / PoinTr 的最优 checkpoint 运行推理，
使用与 eval_ldt.py 完全一致的协议计算评估指标：
  mmd-CD, cov-CD, 1-NN-CD-acc, rec-CD

用法（从 SKD-Net 根目录运行）：
  python baseline/eval_baseline.py --method pcn    --ckpt baseline/PCN/output_mmfi/.../bestl1_network.pth
  python baseline/eval_baseline.py --method mmpoint --ckpt baseline/mmPoint/output_mmfi/.../bestl1_network.pth
  python baseline/eval_baseline.py --method pointr  --ckpt baseline/PoinTr/output_mmfi/.../bestl1_network.pth
"""

import argparse, importlib, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # baseline/
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)                  # SKD-Net/
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from evaluation.evaluation_metrics import compute_CD_metrics, compute_paired_CD

# ── Evaluation protocol (must match SKD-Net) ──────────────────────────────────
DATA_ROOT    = '/home/hjc/Desktop/PRISM/data/MMFi/E01'
VAL_SUBJECTS = ['S08', 'S09', 'S10']
ACTIONS      = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']
CACHE_ROOT   = os.path.join(SCRIPT_DIR, 'mmPoint/data_cache')
LIDAR_NPTS   = 2048
MMWAVE_NPTS  = 512
EVAL_NPTS    = 256    # matches SKD-Net's ref_pts (FPS to 256)

_ACTION_TO_IDX = {a: i for i, a in enumerate(sorted(ACTIONS))}
_ACTION_NAMES  = {v: k for k, v in _ACTION_TO_IDX.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_pc(pc):
    """(B, N, 3) → zero-mean + unit max-radius.  Identical to eval_ldt.py."""
    c = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - c
    r  = torch.amax(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)),
                    dim=1, keepdim=True)
    return pc / (r + 1e-8)


def random_subsample(pc, n):
    """(B, N, 3) → (B, n, 3) random subsample without replacement (or copy if N<n)."""
    N = pc.shape[1]
    if N <= n:
        return pc
    idx = torch.randperm(N, device=pc.device)[:n]
    return pc[:, idx, :]


def _label_from_path(l_path):
    """Extract action label index from a lidar file path."""
    parts = l_path.replace('\\', '/').split('/')
    action = parts[-3]   # e.g. 'A03'
    return _ACTION_TO_IDX.get(action, -1)


# ── Dataset wrapper that adds labels ─────────────────────────────────────────
class LabeledDataset:
    def __init__(self, base_ds):
        self.ds = base_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        l_path = self.ds._records[idx][0]
        item['label'] = _label_from_path(l_path)
        return item


# ── Model loaders ─────────────────────────────────────────────────────────────
def _strip_module(state_dict):
    """Remove 'module.' prefix if present (DataParallel artifact)."""
    if any(k.startswith('module.') for k in state_dict):
        return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def load_pcn(ckpt_path, device):
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PCN'))
    from models.pcn import PCN
    model = PCN(num_coarse=512, num_fine=2048, feat_dim=1024, grid_size=2)
    raw   = torch.load(ckpt_path, map_location=device)
    state = raw.get('net_state_dict', raw)
    model.load_state_dict(_strip_module(state))
    print(f'[PCN] loaded from {ckpt_path}')
    return model.to(device).eval()


def load_mmpoint(ckpt_path, device):
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'mmPoint'))
    import munch, yaml
    args = munch.munchify(yaml.safe_load(
        open(os.path.join(SCRIPT_DIR, 'mmPoint/cfgs/mmPoint_mmfi.yaml'))))
    model_module = importlib.import_module('models.' + args.model_name)
    model = model_module.Model(args)
    raw   = torch.load(ckpt_path, map_location=device)
    state = raw.get('net_state_dict', raw)
    model.load_state_dict(_strip_module(state))
    print(f'[mmPoint] loaded from {ckpt_path}')
    return model.to(device).eval()


def load_pointr(ckpt_path, device):
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PoinTr'))
    from models.pointr import PoinTr
    model = PoinTr(num_centers=128, num_queries=128, k=8,
                   feat_dim=384, num_heads=6,
                   num_enc_layers=6, num_dec_layers=6,
                   num_per_token=16)
    raw   = torch.load(ckpt_path, map_location=device)
    state = raw.get('net_state_dict', raw)
    model.load_state_dict(_strip_module(state))
    print(f'[PoinTr] loaded from {ckpt_path}')
    return model.to(device).eval()


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(method, model, loader, device, template=None):
    all_smp, all_ref, all_lbl = [], [], []

    for batch in tqdm(loader, desc=f'Inference [{method}]'):
        gt  = batch['points'].to(device)   # (B, 2048, 3)
        mmw = batch['image'].to(device)    # (B, 512,  3)
        lbl = batch['label']               # (B,)

        if method == 'pcn':
            _, fine = model(mmw)
            output  = fine                          # (B, 2048, 3)

        elif method == 'mmpoint':
            outs   = model(mmw, template)
            output = outs[2]                        # deform3 (B, 2048, 3)

        elif method == 'pointr':
            _, fine = model(mmw)
            output  = fine                          # (B, 2048, 3)

        smp_sub = random_subsample(output, EVAL_NPTS)  # (B, 256, 3)
        ref_sub = random_subsample(gt,     EVAL_NPTS)  # (B, 256, 3)

        all_smp.append(smp_sub.cpu())
        all_ref.append(ref_sub.cpu())
        all_lbl.append(lbl)

    return (torch.cat(all_smp, 0),
            torch.cat(all_ref, 0),
            torch.cat(all_lbl, 0))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True,
                        choices=['pcn', 'mmpoint', 'pointr'])
    parser.add_argument('--ckpt',   required=True,
                        help='Path to bestl1_network.pth')
    parser.add_argument('--gpu',    type=int, default=0)
    parser.add_argument('--batch',  type=int, default=64)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    # ── Load model ────────────────────────────────────────────────────────────
    loaders = {'pcn': load_pcn, 'mmpoint': load_mmpoint, 'pointr': load_pointr}
    model   = loaders[args.method](args.ckpt, device)

    # ── Load dataset (use PCN's dataset_mmfi which is canonical) ─────────────
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PCN'))
    from dataset.dataset_mmfi import MMFiDataset

    base_ds = MMFiDataset(DATA_ROOT, VAL_SUBJECTS, ACTIONS,
                          lidar_npoints=LIDAR_NPTS,
                          mmwave_npoints=MMWAVE_NPTS,
                          cache_root=CACHE_ROOT)
    ds     = LabeledDataset(base_ds)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=True)

    # ── mmPoint: load human template ─────────────────────────────────────────
    template = None
    if args.method == 'mmpoint':
        sys.path.insert(0, os.path.join(SCRIPT_DIR, 'mmPoint'))
        from utils.model_utils import pc_normalize
        tmpl_path = os.path.join(SCRIPT_DIR,
                                 'mmPoint/human_template/human_template_256.xyz')
        tmpl_np   = np.loadtxt(tmpl_path)
        template  = pc_normalize(tmpl_np, 0.5)

    # ── Inference ─────────────────────────────────────────────────────────────
    smp, ref, labels = run_inference(args.method, model, loader, device, template)

    # ── Normalize (identical to eval_ldt.py) ──────────────────────────────────
    smp = normalize_pc(smp.to(device))
    ref = normalize_pc(ref.to(device))

    N = smp.shape[0]
    print(f'\n总样本数: {N},  每云点数: {EVAL_NPTS}')

    # ── Distribution metrics ──────────────────────────────────────────────────
    print('\n计算分布级 CD 指标 (MMD / COV / 1-NNA)...')
    gen_res = compute_CD_metrics(smp, ref, batch_size=64)

    # ── Paired rec-CD ─────────────────────────────────────────────────────────
    print('\n计算配对 rec-CD...')
    cd_per_sample, rec_cd = compute_paired_CD(smp, ref, batch_size=64)

    # ── Print ─────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'  {args.method.upper()}  ({N} samples, {EVAL_NPTS} pts each)')
    print('=' * 60)
    for k, v in gen_res.items():
        val = v if isinstance(v, float) else v.item()
        print(f'  {k:20s} = {val:.6f}')
    print(f'  {"rec-CD":20s} = {rec_cd:.6f}')

    # Per-action breakdown
    print('\n  -- rec-CD per action --')
    for cls_idx in range(8):
        mask = (labels == cls_idx)
        if mask.sum() > 0:
            mean_cls = cd_per_sample[mask.to(cd_per_sample.device)].mean().item()
            print(f'    {_ACTION_NAMES[cls_idx]} (n={mask.sum():4d}) : {mean_cls:.6f}')
    print('=' * 60)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {k: (v if isinstance(v, float) else v.item())
           for k, v in gen_res.items()}
    out['rec-CD'] = rec_cd
    out_path = os.path.join(SCRIPT_DIR, f'eval_{args.method}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
