"""
vis_teaser.py  ──  PRISM Figure 1 Teaser
=========================================
Default: A27 Bowing

Baseline rows : PCN*  /  PoinTr*  /  SnowflakeNet*  /  mmPoint*   (shown at --baseline_frames)
PRISM rows    : PRISM (Ours)  /  LiDAR GT             (shown at --prism_frames)

The two groups can have completely independent frame sets and column headers,
letting the figure illustrate that baselines are stuck at their frames while
PRISM tracks further in time (or any other temporal contrast of interest).

Coloring logic
--------------
All cells share the same vmin/vmax derived from the first PRISM frame's LiDAR
ground truth (Z = height after unit-sphere normalisation).  Every point is
coloured by its own Z value through the viridis colormap:
  dark blue = low (feet) → yellow-green = high (head)
The shared scale makes cross-cell pose comparisons visually consistent.

Usage
-----
cd ~/Desktop/PRISM
CUDA_VISIBLE_DEVICES=0 python vis_teaser.py \\
    --action A27 \\
    --baseline_frames 30,35,40,45 \\
    --prism_frames    30,35,40,50 \\
    --baseline_frame_labels '$t_1$ (upright),$t_2$ (descent),$t_3$ (bow),$t_4$ (ascent)' \\
    --prism_frame_labels    '$t_1$ (upright),$t_2$ (descent),$t_3$ (bow),$t_5$ (recover)' \\
    --pcn_ckpt        baseline/PCN/output_mmfi/PCN-0327_1055/checkpoints/bestl1_network.pth \\
    --pointr_ckpt     baseline/PoinTr/output_mmfi/PoinTr-0319_1503/checkpoints/bestl1_network.pth \\
    --snowflake_ckpt  baseline/SnowflakeNet/output_mmfi/SnowflakeNet-0327_0113/checkpoints/bestl1_network.pth \\
    --prism_ckpt      experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_3600.pth \\
    --ae_ckpt         experiments/Latent_Diffusion_Trainer/mmfi_202603151358/autoencoder_e_3600.pth \\
    --out             assets/teaser.png
"""

import argparse, glob, os, re, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

DATA_ROOT = 'data/MMFi/E01'
CFG_PATH  = 'experiments/Latent_Diffusion_Trainer/config.yaml'

# ── row groups ────────────────────────────────────────────────────────────────
BASELINE_ROWS       = ['pcn', 'pointr', 'snowflake', 'mmpoint']
BASELINE_ROW_LABELS = ['PCN$^*$', 'PoinTr$^*$', 'SnowflakeNet$^*$', 'mmPoint$^*$']
BASELINE_ROW_BOLD   = [False, False, False, False]

PRISM_ROWS       = ['prism', 'lidar']
PRISM_ROW_LABELS = ['PRISM\n(Ours)', 'LiDAR GT']
PRISM_ROW_BOLD   = [True, False]

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_ACTION          = 'A27'
_DEFAULT_BASELINE_FRAMES = [30, 35, 40, 45]
_DEFAULT_PRISM_FRAMES    = [30, 35, 40, 45]

LIDAR_VIS_N  = 256
MMWAVE_NET_N = 512

ELEV = 18
AZIM = -55
# ── per-method colors (one color per method, consistent with vis_qual_comparison) ─
METHOD_COLORS = {
    'pcn':       '#5B9BD5',   # steel blue
    'pointr':    '#70AD47',   # forest green
    'snowflake': '#ED7D31',   # amber orange
    'mmpoint':   '#9B59B6',   # purple
    'prism':     '#C0392B',   # vivid red  (matches PRISM label color)
    'lidar':     '#2C7BB6',   # steel blue (ground truth)
}

# NOTE: PRISM Score network trained on 8 actions only.
ACTION_MAP = {a: i for i, a in enumerate(sorted(
    ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']))}

CELL_PX  = 320
CELL_DPI = 150
CELL_IN  = CELL_PX / CELL_DPI


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0: return np.zeros((1, 3), np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)

_MM_MAGIC    = bytes([2, 1, 4, 3, 6, 5, 8, 7])
_MM_HDR_FMT  = "<QIIIIIIII"
_MM_HDR_SIZE = 40

def _try_numpy_float32_mm(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0: return None
    if raw.size % 5 == 0:
        pc = raw.reshape(-1, 5)
        if np.all(np.abs(pc[:, :3]) < 20): return pc
    if raw.size % 4 == 0:
        pc = raw.reshape(-1, 4)
        if np.all(np.abs(pc[:, :3]) < 20):
            return np.hstack([pc, np.zeros((pc.shape[0], 1), np.float32)])
    return None

def _parse_ti_uart(path):
    import struct as _s
    raw = open(path, 'rb').read()
    all_pts, pos = [], 0
    while True:
        idx = raw.find(_MM_MAGIC, pos)
        if idx == -1: break
        if idx + _MM_HDR_SIZE > len(raw): break
        total_len = _s.unpack_from('<I', raw, idx + 12)[0]
        frame = raw[idx:idx + total_len]
        if len(frame) < _MM_HDR_SIZE: pos = idx + _MM_HDR_SIZE; continue
        try: header = _s.unpack_from(_MM_HDR_FMT, frame, 0)
        except: pos = idx + _MM_HDR_SIZE; continue
        if header[0] != _s.unpack('<Q', _MM_MAGIC)[0]: pos = idx + _MM_HDR_SIZE; continue
        pts, offset = [], _MM_HDR_SIZE
        for _ in range(header[7]):
            if offset + 8 > len(frame): break
            tlv_t, tlv_l = _s.unpack_from('<II', frame, offset); offset += 8
            if tlv_t == 1:
                for _ in range(tlv_l // 16):
                    if offset + 16 > len(frame): break
                    x, y, z, v = _s.unpack_from('<ffff', frame, offset)
                    pts.append([x, y, z, v, 0.]); offset += 16
            elif tlv_t == 7:
                for _ in range(tlv_l // 20):
                    if offset + 20 > len(frame): break
                    x, y, z, v = _s.unpack_from('<ffff', frame, offset)
                    snr, _ = _s.unpack_from('<HH', frame, offset + 16)
                    pts.append([x, y, z, v, float(snr)]); offset += 20
            else: offset += tlv_l
        if pts: all_pts.append(np.array(pts, np.float32))
        pos = max(idx + _MM_HDR_SIZE, idx + total_len)
    return np.vstack(all_pts) if all_pts else None

def _load_mmwave(path):
    for fn in [_try_numpy_float32_mm, _parse_ti_uart]:
        pc = fn(path)
        if pc is not None and len(pc) > 0: return pc[:, :3]
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size % 5 == 0 and raw64.size > 0:
        return raw64.reshape(-1, 5).astype(np.float32)[:, :3]
    return np.zeros((1, 3), np.float32)

def _normalize(pc):
    c = pc.mean(0, keepdims=True)
    d = pc - c
    s = max(np.sqrt((d ** 2).sum(1)).max(), 1e-8)
    return (d / s).astype(np.float32), c.astype(np.float32), float(s)

def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0: return np.zeros((n, 3), np.float32)
    return pc[np.random.choice(N, n, replace=N < n)]

def _norm256(pc):
    c = pc.mean(0, keepdims=True); d = pc - c
    s = max(np.sqrt((d ** 2).sum(1)).max(), 1e-8)
    return _subsample(d / s, 256)


def load_frame_at(action, frame_idx, subject):
    """
    Load data at a specific 0-based sorted frame index for (action, subject).
    Returns dict with lidar_vis, mm_vis512, skel_norm, frame_num.
    """
    l_dir   = os.path.join(DATA_ROOT, subject, action, 'lidar')
    m_dir   = os.path.join(DATA_ROOT, subject, action, 'mmwave')
    gt_path = os.path.join(DATA_ROOT, subject, action, 'ground_truth.npy')

    gt = np.load(gt_path, allow_pickle=True)
    if gt.dtype == object: gt = np.array(gt.tolist(), np.float32)
    else: gt = gt.astype(np.float32)
    s = gt.shape
    if   gt.ndim == 3 and s[1] == 17 and s[2] == 3: pass
    elif gt.ndim == 3 and s[1] == 3  and s[2] == 17: gt = gt.transpose(0, 2, 1)
    elif gt.ndim == 2 and s[1] == 51:                gt = gt.reshape(s[0], 17, 3)

    lf = {_frame_num(f): f for f in glob.glob(os.path.join(l_dir, 'frame*_filtered.bin'))}
    mf = {_frame_num(f): f for f in glob.glob(os.path.join(m_dir, 'frame*_filtered.bin'))}
    lf.pop(-1, None); mf.pop(-1, None)
    common = sorted(set(lf) & set(mf)); common = [fn for fn in common if fn - 1 < len(gt)]

    fn = common[min(frame_idx, len(common) - 1)]
    print(f'  [{action}] idx {frame_idx:3d} → frame #{fn}')

    lidar_raw = _load_lidar(lf[fn])
    lidar_n, shift, scale = _normalize(lidar_raw)
    lidar_vis = _subsample(lidar_n, LIDAR_VIS_N)

    mm = _load_mmwave(mf[fn])
    mm = mm[np.isfinite(mm).all(1) & (np.abs(mm) < 20).all(1)]
    if mm.shape[0] == 0: mm = np.zeros((1, 3), np.float32)
    mm_n, _, _ = _normalize(mm)

    skel_n = ((gt[fn - 1][:, [2, 0, 1]] - shift) / scale).astype(np.float32)

    return dict(
        lidar_vis = lidar_vis,
        mm_vis512 = _subsample(mm_n, MMWAVE_NET_N),
        skel_norm = skel_n,
        frame_num = fn,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def _strip_module(sd):
    if any(k.startswith('module.') for k in sd):
        return {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd

def _swap_path(new_dir):
    for d in [os.path.join(ROOT_DIR, 'baseline', 'PCN'),
              os.path.join(ROOT_DIR, 'baseline', 'PoinTr'),
              os.path.join(ROOT_DIR, 'baseline', 'SnowflakeNet'),
              os.path.join(ROOT_DIR, 'baseline', 'mmPoint')]:
        while d in sys.path: sys.path.remove(d)
    sys.path.insert(0, new_dir)
    for k in [k for k in sys.modules if k == 'models' or k.startswith('models.')]:
        del sys.modules[k]

def load_pcn(ckpt, device):
    _swap_path(os.path.join(ROOT_DIR, 'baseline', 'PCN'))
    from models.pcn import PCN
    m = PCN(num_coarse=512, num_fine=2048, feat_dim=1024, grid_size=2)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(_strip_module(sd.get('net_state_dict', sd)))
    print(f'[PCN]          {ckpt}'); return m.to(device).eval()

def load_pointr(ckpt, device):
    _swap_path(os.path.join(ROOT_DIR, 'baseline', 'PoinTr'))
    from models.pointr import PoinTr
    m = PoinTr(num_centers=128, num_queries=128, k=8,
               feat_dim=384, num_heads=6,
               num_enc_layers=6, num_dec_layers=6, num_per_token=16)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(_strip_module(sd.get('net_state_dict', sd)))
    print(f'[PoinTr]       {ckpt}'); return m.to(device).eval()

def load_snowflake(ckpt, device):
    _swap_path(os.path.join(ROOT_DIR, 'baseline', 'SnowflakeNet'))
    from models.snowflakenet import SnowflakeNet
    m = SnowflakeNet(dim_feat=512, num_pc=256, num_p0=512,
                     radius=1, bounding=True, up_factors=[2, 2])
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(_strip_module(sd.get('net_state_dict', sd)))
    print(f'[SnowflakeNet] {ckpt}'); return m.to(device).eval()

def load_mmpoint(ckpt, device):
    _swap_path(os.path.join(ROOT_DIR, 'baseline', 'mmPoint'))
    import munch, yaml as _yaml
    cfg_path = os.path.join(ROOT_DIR, 'baseline', 'mmPoint', 'cfgs', 'mmPoint_mmfi.yaml')
    args = munch.munchify(_yaml.safe_load(open(cfg_path)))
    import importlib
    model_module = importlib.import_module('models.' + args.model_name)
    m = model_module.Model(args)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(_strip_module(sd.get('net_state_dict', sd)))
    # load and normalise the human template (required at inference time)
    from utils.model_utils import pc_normalize
    tmpl_np = np.loadtxt(os.path.join(ROOT_DIR, 'baseline', 'mmPoint',
                                      'human_template', 'human_template_256.xyz'))
    # keep template as numpy — mmPoint.forward() calls torch.FloatTensor(...).cuda() internally
    template = pc_normalize(tmpl_np, 0.5)
    print(f'[mmPoint]      {ckpt}')
    return m.to(device).eval(), template

def load_prism(cfg_path, ckpt, ae_ckpt, device):
    import yaml
    from model.scorenet.score import Score
    from model.Compressor.Network import Compressor
    from model.GeoAE.Network import GeoAE
    from completion_trainer.Latent_SDE_Trainer import Trainer
    from tools.io import dict2namespace
    with open(cfg_path) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader))
    trainer = Trainer(cfg, model=Score(cfg.score),
                      compressor=Compressor(cfg.compressor), device=device)
    ck = torch.load(ckpt, map_location='cpu')
    trainer.model.load_state_dict(ck['score_state_dict'], strict=True)
    trainer.compressor.load_state_dict(ck['compressor_state_dict'], strict=True)
    trainer.compressor.init()
    if hasattr(trainer.optimizer, 'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    trainer.model.eval(); trainer.compressor.eval()
    ae_cfg = getattr(cfg, 'geo_ae', None)
    ae = GeoAE(cfg=ae_cfg,
               feat_dim=getattr(ae_cfg, 'feat_dim', 256) if ae_cfg else 256,
               num_coarse=getattr(ae_cfg, 'num_coarse', 128) if ae_cfg else 128).to(device)
    if ae_ckpt and os.path.exists(ae_ckpt):
        ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
    ae.eval(); print(f'[PRISM]        epoch={ck.get("epoch", "?")}')
    return trainer, ae, cfg


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_pcn(model, mm512, device):
    _, fine = model(torch.from_numpy(mm512).float().unsqueeze(0).to(device))
    return fine[0].cpu().numpy()

@torch.no_grad()
def infer_pointr(model, mm512, device):
    _, fine = model(torch.from_numpy(mm512).float().unsqueeze(0).to(device))
    return fine[0].cpu().numpy()

@torch.no_grad()
def infer_snowflake(model, mm512, device):
    preds = model(torch.from_numpy(mm512).float().unsqueeze(0).to(device))
    return preds[-1][0].cpu().numpy()

@torch.no_grad()
def infer_mmpoint(model, template, mm512, device):
    # template is a numpy array; mmPoint.forward() calls .cuda() on it internally
    outs = model(torch.from_numpy(mm512).float().unsqueeze(0).to(device), template)
    return outs[2][0].cpu().numpy()   # deform3: finest output (2048, 3)

@torch.no_grad()
def infer_prism(trainer, ae, fd, action, cfg, device):
    mm   = torch.from_numpy(fd['mm_vis512']).float().unsqueeze(0).to(device)
    skel = torch.from_numpy(fd['skel_norm']).unsqueeze(0).to(device)
    lbl  = torch.tensor([ACTION_MAP[action]], dtype=torch.long).to(device)
    coarse, _ = ae(mm)
    if hasattr(trainer.optimizer, 'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    zs = (cfg.score.z_scale,
          cfg.score.z_dim + 3 if cfg.score.graphconv else cfg.score.z_dim)
    z_list = trainer.SDE.sample_discrete(
        score_fn=trainer.score_fn, num_samples=1, N=cfg.sde.sample_N,
        corrector=cfg.sde.corrector, predictor=cfg.sde.predictor,
        corrector_steps=cfg.sde.corrector_steps, shape=zs,
        time_eps=trainer.sample_time_eps, label=lbl,
        denoise=cfg.sde.denoise, device=device,
        probability_flow=cfg.sde.probability_flow, snr=cfg.sde.snr,
        condition={'skeleton': skel, 'coarse': coarse},
        print_steps=cfg.sde.sample_N)
    pc = trainer.compressor.sample((1, cfg.common.num_points), given_eps=z_list[-1])
    if hasattr(trainer.optimizer, 'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    return pc[0].cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Cell rendering
# ══════════════════════════════════════════════════════════════════════════════

_RENDER_STYLE = {
    'pcn':       dict(s=5, alpha=0.90, depthshade=True),
    'pointr':    dict(s=5, alpha=0.90, depthshade=True),
    'snowflake': dict(s=5, alpha=0.90, depthshade=True),
    'mmpoint':   dict(s=5, alpha=0.90, depthshade=True),
    'prism':     dict(s=5, alpha=0.97, depthshade=True),
    'lidar':     dict(s=5, alpha=0.95, depthshade=True),
}

def _cell_img(pc, ref_pc, row_key):
    """Render one 3-D scatter cell; ref_pc fixes axis bounds only."""
    fig = plt.figure(figsize=(CELL_IN, CELL_IN), facecolor='white', dpi=CELL_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # axis bounds from reference LiDAR GT (consistent across all cells)
    mn, mx = ref_pc.min(0), ref_pc.max(0)
    mid = (mn + mx) / 2
    r   = max((mx - mn).max() / 2 * 0.92, 0.3)

    ax.set_axis_off()
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)

    style = _RENDER_STYLE.get(row_key, dict(s=3, alpha=0.90, depthshade=True))
    color = METHOD_COLORS.get(row_key, '#888888')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               c=color, s=style['s'], alpha=style['alpha'],
               depthshade=style['depthshade'], linewidths=0)

    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf, dtype=np.uint8).reshape(CELL_PX, CELL_PX, 4).copy()
    plt.close(fig)
    return img

def _white_bg(rgba):
    rgb = rgba[:, :, :3].astype(np.float32) / 255.
    a   = rgba[:, :, 3:4].astype(np.float32) / 255.
    return ((rgb * a + (1. - a)) * 255).astype(np.uint8)

def _autocrop(img, pad=14):
    """Trim white borders from a cell, keeping `pad` px margin on each side."""
    mask = np.any(img < 245, axis=2)   # any non-white pixel
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0 = max(rows[0]  - pad, 0);  r1 = min(rows[-1]  + pad + 1, img.shape[0])
    c0 = max(cols[0]  - pad, 0);  c1 = min(cols[-1]  + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]

def _pad_group(cells):
    """Pad all cells in a group (list of rows × cols) to the same max size."""
    max_h = max(c.shape[0] for row in cells for c in row)
    max_w = max(c.shape[1] for row in cells for c in row)
    out = []
    for row in cells:
        r = []
        for c in row:
            h, w = c.shape[:2]
            canvas = np.full((max_h, max_w, 3), 255, np.uint8)
            y0 = (max_h - h) // 2; x0 = (max_w - w) // 2
            canvas[y0:y0+h, x0:x0+w] = c
            r.append(canvas)
        out.append(r)
    return out, max_h, max_w


# ══════════════════════════════════════════════════════════════════════════════
#  Assembly  (two independent panels stacked vertically)
# ══════════════════════════════════════════════════════════════════════════════

def assemble_and_save(baseline_cells, prism_cells,
                      baseline_labels, prism_labels,
                      out_path, dpi):
    """
    TRANSPOSED LAYOUT — methods as columns, frames as rows.
    Wider than tall → suitable for NeurIPS single-column.

    baseline_cells : list[n_b_methods][n_b_frames] → uint8 RGB
    prism_cells    : list[n_p_methods][n_p_frames] → uint8 RGB

    Left panel : n_b_frames rows × n_b_methods cols  (baselines)
    Right panel: n_p_frames rows × n_p_methods cols  (PRISM + GT)
    Vertical separator between panels.

    Column labels (method names) → top of figure.
    Row labels (frame labels)    → left of figure (horizontal text).
    PRISM frame labels           → right of figure when they differ from baseline.
    """
    b_cells, b_cell_h, b_cell_w = _pad_group(baseline_cells)
    p_cells, p_cell_h, p_cell_w = _pad_group(prism_cells)

    # unify cell size across both groups
    cell_h = max(b_cell_h, p_cell_h)
    cell_w = max(b_cell_w, p_cell_w)

    def _resize_group(cells, ch, cw):
        out = []
        for row in cells:
            r = []
            for c in row:
                canvas = np.full((ch, cw, 3), 255, np.uint8)
                h, w = c.shape[:2]
                y0 = (ch - h) // 2; x0 = (cw - w) // 2
                canvas[y0:y0+h, x0:x0+w] = c
                r.append(canvas)
            out.append(r)
        return out

    b_cells = _resize_group(b_cells, cell_h, cell_w)
    p_cells = _resize_group(p_cells, cell_h, cell_w)

    # current shape: cells[method_idx][frame_idx]
    # transpose to:  cells_T[frame_idx][method_idx]  (frames become rows)
    n_b_methods = len(b_cells);  n_b_frames = len(b_cells[0]) if b_cells else 0
    n_p_methods = len(p_cells);  n_p_frames = len(p_cells[0]) if p_cells else 0

    b_T = [[b_cells[m][f] for m in range(n_b_methods)] for f in range(n_b_frames)]
    p_T = [[p_cells[m][f] for m in range(n_p_methods)] for f in range(n_p_frames)]

    # pad shorter panel with blank rows so both have same height
    n_frames = max(n_b_frames, n_p_frames)
    blank_b = [np.full((cell_h, cell_w, 3), 255, np.uint8)] * n_b_methods
    blank_p = [np.full((cell_h, cell_w, 3), 255, np.uint8)] * n_p_methods
    while len(b_T) < n_frames: b_T.append(blank_b)
    while len(p_T) < n_frames: p_T.append(blank_p)

    b_grid = np.vstack([np.hstack(row) for row in b_T])   # (n_frames*ch, n_b_methods*cw)
    p_grid = np.vstack([np.hstack(row) for row in p_T])   # (n_frames*ch, n_p_methods*cw)

    # vertical separator strip (thin grey line)
    sep_px   = max(cell_w // 20, 5)
    sep_h_px = n_frames * cell_h
    sep_strip = np.full((sep_h_px, sep_px, 3), 255, np.uint8)
    lx = sep_px // 2
    sep_strip[:, lx - 1:lx + 2] = [185, 185, 185]

    grid = np.hstack([b_grid, sep_strip, p_grid])   # landscape: wide × short
    grid_h_px, grid_w_px = grid.shape[:2]
    grid_h_in = grid_h_px / dpi
    grid_w_in = grid_w_px / dpi

    # ── figure geometry ───────────────────────────────────────────────────────
    margin_left  = 0.70   # for frame row labels (horizontal text)
    margin_top   = 0.30   # for method column labels
    margin_right = 0.0    # extended below if PRISM labels differ

    # check if we need a right margin for PRISM row labels
    show_prism_row_labels = (prism_labels != baseline_labels)
    if show_prism_row_labels:
        margin_right = 0.68

    fig_w = margin_left + grid_w_in + margin_right
    fig_h = margin_top  + grid_h_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='white')

    ax = fig.add_axes([margin_left / fig_w, 0.0,
                       grid_w_in   / fig_w, grid_h_in / fig_h])
    ax.imshow(grid, aspect='equal', interpolation='lanczos')
    ax.set_axis_off()

    # ── coordinate helpers ────────────────────────────────────────────────────
    def _xfig(x_px):
        """Grid pixel x (from left of grid) → figure x."""
        return (margin_left + x_px / dpi) / fig_w

    def _yfig(y_px):
        """Grid pixel y (from top of grid) → figure y (bottom=0)."""
        return (grid_h_in - y_px / dpi) / fig_h

    # ── column labels: baseline methods (left panel, top) ────────────────────
    y_col_lbl = _yfig(0) + margin_top * 0.55 / fig_h
    for j, (label, bold) in enumerate(zip(BASELINE_ROW_LABELS, BASELINE_ROW_BOLD)):
        x = _xfig((j + 0.5) * cell_w)
        color  = '#c0392b' if bold else '#111111'
        weight = 'bold'    if bold else 'semibold'
        fig.text(x, y_col_lbl, label.replace('\n', ' '),
                 ha='center', va='center',
                 fontsize=5.5, fontweight=weight, color=color,
                 transform=fig.transFigure)

    # ── column labels: PRISM+GT (right panel, top) ───────────────────────────
    x_p0 = n_b_methods * cell_w + sep_px          # pixel x of right panel start
    for j, (label, bold) in enumerate(zip(PRISM_ROW_LABELS, PRISM_ROW_BOLD)):
        x = _xfig(x_p0 + (j + 0.5) * cell_w)
        color  = '#c0392b' if bold else '#111111'
        weight = 'bold'    if bold else 'semibold'
        fig.text(x, y_col_lbl, label.replace('\n', ' '),
                 ha='center', va='center',
                 fontsize=5.5, fontweight=weight, color=color,
                 transform=fig.transFigure)

    # ── row labels: baseline frame labels (left side, horizontal) ────────────
    x_row_lbl = margin_left * 0.45 / fig_w
    for i, label in enumerate(baseline_labels):
        y = _yfig((i + 0.5) * cell_h)
        fig.text(x_row_lbl, y, label,
                 ha='center', va='center',
                 fontsize=5.5, fontweight='normal', color='#333333',
                 transform=fig.transFigure)

    # ── row labels: PRISM frame labels (right side, only when different) ─────
    if show_prism_row_labels:
        x_prism_lbl = (margin_left + grid_w_in + margin_right * 0.55) / fig_w
        for i, label in enumerate(prism_labels):
            if i < n_p_frames:
                y = _yfig((i + 0.5) * cell_h)
                fig.text(x_prism_lbl, y, label,
                         ha='center', va='center',
                         fontsize=6.0, fontweight='normal', color='#555555',
                         transform=fig.transFigure)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor='white',
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'\n✓  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_frames(s):
    return [int(x.strip()) for x in s.split(',')]

def _default_labels(n):
    return [f'$t_{{{i+1}}}$' for i in range(n)]

def main():
    parser = argparse.ArgumentParser(
        description='PRISM teaser: temporal coherence figure with split frame sets')

    # ── action / subject ──────────────────────────────────────────────────────
    parser.add_argument('--action',  default=_DEFAULT_ACTION,
        help=f'Action code (must be in 8-action set: {sorted(ACTION_MAP)})')
    parser.add_argument('--subject', default='S01')

    # ── independent frame sets ────────────────────────────────────────────────
    parser.add_argument('--baseline_frames',
        default=','.join(map(str, _DEFAULT_BASELINE_FRAMES)),
        help='Comma-separated 0-based frame indices for PCN/PoinTr/SnowflakeNet rows')
    parser.add_argument('--prism_frames',
        default=','.join(map(str, _DEFAULT_PRISM_FRAMES)),
        help='Comma-separated 0-based frame indices for PRISM and LiDAR GT rows')

    parser.add_argument('--baseline_frame_labels', default=None,
        help='Comma-separated LaTeX column headers for the baseline group '
             '(default: $t_1$, $t_2$, …)')
    parser.add_argument('--prism_frame_labels', default=None,
        help='Comma-separated LaTeX column headers for the PRISM/GT group '
             '(default: $t_1$, $t_2$, …)')

    # ── checkpoints ───────────────────────────────────────────────────────────
    parser.add_argument('--pcn_ckpt',
        default='baseline/PCN/output_mmfi/PCN-0327_1055/checkpoints/bestl1_network.pth')
    parser.add_argument('--pointr_ckpt',
        default='baseline/PoinTr/output_mmfi/PoinTr-0319_1503/checkpoints/bestl1_network.pth')
    parser.add_argument('--snowflake_ckpt',
        default='baseline/SnowflakeNet/output_mmfi/SnowflakeNet-0327_0113/checkpoints/bestl1_network.pth')
    parser.add_argument('--mmpoint_ckpt',
        default='baseline/mmPoint/output_mmfi/mmPoint_mmfi-0319_1339/checkpoints/bestl1_network.pth')
    parser.add_argument('--prism_ckpt',
        default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_3600.pth')
    parser.add_argument('--ae_ckpt',
        default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/autoencoder_e_3600.pth')
    parser.add_argument('--cfg',  default=CFG_PATH)

    # ── output ────────────────────────────────────────────────────────────────
    parser.add_argument('--out',  default='assets/teaser.png')
    parser.add_argument('--dpi',  type=int, default=200)
    parser.add_argument('--gpu',  type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ── validate action ───────────────────────────────────────────────────────
    assert args.action in ACTION_MAP, \
        f'Action {args.action} not in 8-action set: {sorted(ACTION_MAP)}'

    baseline_frame_indices = _parse_frames(args.baseline_frames)
    prism_frame_indices    = _parse_frames(args.prism_frames)

    n_b = len(baseline_frame_indices)
    n_p = len(prism_frame_indices)

    baseline_labels = (args.baseline_frame_labels.split(',')
                       if args.baseline_frame_labels else _default_labels(n_b))
    prism_labels    = (args.prism_frame_labels.split(',')
                       if args.prism_frame_labels    else _default_labels(n_p))

    assert len(baseline_labels) == n_b, \
        f'--baseline_frame_labels must have {n_b} entries'
    assert len(prism_labels) == n_p, \
        f'--prism_frame_labels must have {n_p} entries'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Action : {args.action}  Subject: {args.subject}')
    print(f'Baseline frames : {baseline_frame_indices}')
    print(f'PRISM frames    : {prism_frame_indices}')

    # ── load models ───────────────────────────────────────────────────────────
    print('\n[1] Loading models...')
    pcn                 = load_pcn(args.pcn_ckpt, device)
    pointr              = load_pointr(args.pointr_ckpt, device)
    snowflake           = load_snowflake(args.snowflake_ckpt, device)
    mmpoint, mm_tmpl    = load_mmpoint(args.mmpoint_ckpt, device)
    trainer, ae, cfg    = load_prism(args.cfg, args.prism_ckpt, args.ae_ckpt, device)

    # ── load all required frames (union) ─────────────────────────────────────
    all_indices = sorted(set(baseline_frame_indices) | set(prism_frame_indices))
    print(f'\n[2] Loading frames {all_indices} of {args.action} ({args.subject})...')
    frames = {}
    for fi in all_indices:
        frames[fi] = load_frame_at(args.action, fi, args.subject)

    # shared reference for consistent axis bounds and colour scale
    ref_pc = frames[prism_frame_indices[0]]['lidar_vis']

    # ── inference: baselines ──────────────────────────────────────────────────
    print('\n[3] Baseline inference...')
    raw_b = {}
    for fi in baseline_frame_indices:
        fd    = frames[fi]
        mm512 = fd['mm_vis512']
        print(f'  frame {fi}: PCN', end='', flush=True)
        pc_pcn = infer_pcn(pcn, mm512, device)
        print(' | PoinTr', end='', flush=True)
        pc_ptr = infer_pointr(pointr, mm512, device)
        print(' | SnowflakeNet', end='', flush=True)
        pc_sfl = infer_snowflake(snowflake, mm512, device)
        print(' | mmPoint', end='', flush=True)
        pc_mmp = infer_mmpoint(mmpoint, mm_tmpl, mm512, device)
        print()
        raw_b[fi] = dict(
            pcn       = _norm256(pc_pcn),
            pointr    = _norm256(pc_ptr),
            snowflake = _norm256(pc_sfl),
            mmpoint   = _norm256(pc_mmp),
        )

    # ── inference: PRISM ──────────────────────────────────────────────────────
    print('\n[4] PRISM inference...')
    raw_p = {}
    for fi in prism_frame_indices:
        fd = frames[fi]
        print(f'  frame {fi}: PRISM...', flush=True)
        pc_pr = infer_prism(trainer, ae, fd, args.action, cfg, device)
        raw_p[fi] = dict(
            prism = pc_pr,
            lidar = fd['lidar_vis'],
        )

    # ── render cells ──────────────────────────────────────────────────────────
    print('\n[5] Rendering cells...')
    total = len(BASELINE_ROWS) * n_b + len(PRISM_ROWS) * n_p
    k = 0

    baseline_cells = []
    for rk in BASELINE_ROWS:
        row = []
        for fi in baseline_frame_indices:
            rgba = _cell_img(raw_b[fi][rk], ref_pc, rk)
            row.append(_autocrop(_white_bg(rgba)))
            k += 1; sys.stdout.write(f'\r  {k}/{total}'); sys.stdout.flush()
        baseline_cells.append(row)

    prism_cells = []
    for rk in PRISM_ROWS:
        row = []
        for fi in prism_frame_indices:
            rgba = _cell_img(raw_p[fi][rk], ref_pc, rk)
            row.append(_autocrop(_white_bg(rgba)))
            k += 1; sys.stdout.write(f'\r  {k}/{total}'); sys.stdout.flush()
        prism_cells.append(row)
    print()

    # ── assemble ──────────────────────────────────────────────────────────────
    print('[6] Assembling figure...')
    assemble_and_save(baseline_cells, prism_cells,
                      baseline_labels, prism_labels,
                      args.out, args.dpi)
    print(f'    Open with: eog {args.out}')


if __name__ == '__main__':
    main()
