"""
vis_qual_comparison.py  ──  PRISM Figure 6 : Qualitative Comparison
====================================================================
Rows : PCN*  /  PoinTr*  /  SnowflakeNet*  /  PRISM (Ours)  /  LiDAR GT
Cols : A27 Bowing  /  A17 Wave Hand  /  A22 Kick  /  A12 Squat

Usage
-----
python vis_qual_comparison.py \\
    --prism_ckpt  experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_3600.pth \\
    --ae_ckpt     experiments/Latent_Diffusion_Trainer/mmfi_202603151358/autoencoder_e_3600.pth \\
    --out         assets/qual_comparison.png
"""

import argparse, glob, io, os, re, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

DATA_ROOT = 'data/MMFi/E01'
CFG_PATH  = 'experiments/Latent_Diffusion_Trainer/config.yaml'
SUBJECT   = 'S01'

ACTIONS     = ['A27', 'A17', 'A22', 'A12']
FRAME_IDX   = {'A27': 51, 'A17': 44, 'A22': 50, 'A12': 8}
ACTION_NAMES= {'A27': 'Bowing', 'A17': 'Wave Hand', 'A22': 'Kick', 'A12': 'Squat'}

MMWAVE_VIS_N = 256
MMWAVE_NET_N = 512
LIDAR_VIS_N  = 256

ELEV = 18
AZIM = -55


# ══════════════════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════════════════

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0: return np.zeros((1,3), np.float32)
    return raw[:(raw.size//3)*3].reshape(-1,3).astype(np.float32)

_MM_MAGIC    = bytes([2,1,4,3,6,5,8,7])
_MM_HDR_FMT  = "<QIIIIIIII"
_MM_HDR_SIZE = 40

def _try_numpy_float32_mm(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0: return None
    if raw.size%5==0:
        pc = raw.reshape(-1,5)
        if np.all(np.abs(pc[:,:3])<20): return pc
    if raw.size%4==0:
        pc = raw.reshape(-1,4)
        if np.all(np.abs(pc[:,:3])<20):
            return np.hstack([pc, np.zeros((pc.shape[0],1),np.float32)])
    return None

def _parse_ti_uart(path):
    import struct as _s
    raw = open(path,'rb').read()
    all_pts, pos = [], 0
    while True:
        idx = raw.find(_MM_MAGIC, pos)
        if idx==-1: break
        if idx+_MM_HDR_SIZE>len(raw): break
        total_len = _s.unpack_from('<I',raw,idx+12)[0]
        frame = raw[idx:idx+total_len]
        if len(frame)<_MM_HDR_SIZE: pos=idx+_MM_HDR_SIZE; continue
        try: header = _s.unpack_from(_MM_HDR_FMT,frame,0)
        except: pos=idx+_MM_HDR_SIZE; continue
        if header[0]!=_s.unpack('<Q',_MM_MAGIC)[0]: pos=idx+_MM_HDR_SIZE; continue
        pts, offset = [], _MM_HDR_SIZE
        for _ in range(header[7]):
            if offset+8>len(frame): break
            tlv_t,tlv_l = _s.unpack_from('<II',frame,offset); offset+=8
            if tlv_t==1:
                for _ in range(tlv_l//16):
                    if offset+16>len(frame): break
                    x,y,z,v=_s.unpack_from('<ffff',frame,offset)
                    pts.append([x,y,z,v,0.]); offset+=16
            elif tlv_t==7:
                for _ in range(tlv_l//20):
                    if offset+20>len(frame): break
                    x,y,z,v=_s.unpack_from('<ffff',frame,offset)
                    snr,_=_s.unpack_from('<HH',frame,offset+16)
                    pts.append([x,y,z,v,float(snr)]); offset+=20
            else: offset+=tlv_l
        if pts: all_pts.append(np.array(pts,np.float32))
        pos=max(idx+_MM_HDR_SIZE, idx+total_len)
    return np.vstack(all_pts) if all_pts else None

def _load_mmwave(path):
    for fn in [_try_numpy_float32_mm, _parse_ti_uart]:
        pc = fn(path)
        if pc is not None and len(pc)>0: return pc[:,:3]
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size%5==0 and raw64.size>0:
        return raw64.reshape(-1,5).astype(np.float32)[:,:3]
    return np.zeros((1,3),np.float32)

def _normalize(pc):
    c = pc.mean(0, keepdims=True)
    d = pc - c
    s = max(np.sqrt((d**2).sum(1)).max(), 1e-8)
    return (d/s).astype(np.float32), c.astype(np.float32), float(s)

def _subsample(pc, n):
    N = pc.shape[0]
    if N==0: return np.zeros((n,3),np.float32)
    return pc[np.random.choice(N, n, replace=N<n)]

def _norm256(pc):
    c = pc.mean(0, keepdims=True); d = pc-c
    s = max(np.sqrt((d**2).sum(1)).max(), 1e-8)
    return _subsample(d/s, 256)

def load_frame(action):
    fi = FRAME_IDX[action]
    l_dir = os.path.join(DATA_ROOT, SUBJECT, action, 'lidar')
    m_dir = os.path.join(DATA_ROOT, SUBJECT, action, 'mmwave')
    gt_path = os.path.join(DATA_ROOT, SUBJECT, action, 'ground_truth.npy')

    gt = np.load(gt_path, allow_pickle=True)
    if gt.dtype==object: gt = np.array(gt.tolist(), np.float32)
    else: gt = gt.astype(np.float32)
    s = gt.shape
    if   gt.ndim==3 and s[1]==17 and s[2]==3: pass
    elif gt.ndim==3 and s[1]==3  and s[2]==17: gt = gt.transpose(0,2,1)
    elif gt.ndim==2 and s[1]==51: gt = gt.reshape(s[0],17,3)
    else: raise ValueError(f'GT shape: {gt.shape}')

    lf = {_frame_num(f): f for f in glob.glob(os.path.join(l_dir,'frame*_filtered.bin'))}
    mf = {_frame_num(f): f for f in glob.glob(os.path.join(m_dir,'frame*_filtered.bin'))}
    lf.pop(-1,None); mf.pop(-1,None)
    common = sorted(set(lf)&set(mf)); common=[fn for fn in common if fn-1<len(gt)]
    fn = common[min(fi, len(common)-1)]
    print(f'  [{action}] frame #{fn}')

    lidar_raw = _load_lidar(lf[fn])
    lidar_n, shift, scale = _normalize(lidar_raw)
    lidar_vis = _subsample(lidar_n, LIDAR_VIS_N)

    mm = _load_mmwave(mf[fn])
    mm = mm[np.isfinite(mm).all(1) & (np.abs(mm)<20).all(1)]
    if mm.shape[0]==0: mm = np.zeros((1,3),np.float32)
    mm_n,_,_ = _normalize(mm)

    skel_n = ((gt[fn-1][:,[2,0,1]] - shift) / scale).astype(np.float32)

    return dict(
        lidar_vis  = lidar_vis,
        mm_vis256  = _subsample(mm_n, MMWAVE_VIS_N),
        mm_vis512  = _subsample(mm_n, MMWAVE_NET_N),
        skel_norm  = skel_n,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  模型加载
# ══════════════════════════════════════════════════════════════════════════════

def _strip_module(sd):
    if any(k.startswith('module.') for k in sd):
        return {k.replace('module.','',1):v for k,v in sd.items()}
    return sd

def _swap_path(new_dir):
    for d in [os.path.join(ROOT_DIR,'baseline','PCN'),
              os.path.join(ROOT_DIR,'baseline','PoinTr'),
              os.path.join(ROOT_DIR,'baseline','SnowflakeNet'),
              os.path.join(ROOT_DIR,'baseline','mmPoint')]:
        while d in sys.path: sys.path.remove(d)
    sys.path.insert(0, new_dir)
    for k in [k for k in sys.modules if k=='models' or k.startswith('models.')]:
        del sys.modules[k]

def load_pcn(ckpt, device):
    _swap_path(os.path.join(ROOT_DIR,'baseline','PCN'))
    from models.pcn import PCN
    m = PCN(num_coarse=512, num_fine=2048, feat_dim=1024, grid_size=2)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(_strip_module(sd.get('net_state_dict',sd)))
    print(f'[PCN]    {ckpt}'); return m.to(device).eval()

def load_pointr(ckpt, device):
    _swap_path(os.path.join(ROOT_DIR,'baseline','PoinTr'))
    from models.pointr import PoinTr
    m = PoinTr(num_centers=128, num_queries=128, k=8,
               feat_dim=384, num_heads=6,
               num_enc_layers=6, num_dec_layers=6, num_per_token=16)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(_strip_module(sd.get('net_state_dict',sd)))
    print(f'[PoinTr] {ckpt}'); return m.to(device).eval()

def load_snowflake(ckpt, device):
    _swap_path(os.path.join(ROOT_DIR,'baseline','SnowflakeNet'))
    from models.snowflakenet import SnowflakeNet
    m = SnowflakeNet(dim_feat=512, num_pc=256, num_p0=512,
                     radius=1, bounding=True, up_factors=[2,2])
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(_strip_module(sd.get('net_state_dict',sd)))
    print(f'[Snowflake] {ckpt}'); return m.to(device).eval()

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
    from utils.model_utils import pc_normalize
    tmpl_np = np.loadtxt(os.path.join(ROOT_DIR, 'baseline', 'mmPoint',
                                      'human_template', 'human_template_256.xyz'))
    template = pc_normalize(tmpl_np, 0.5)
    print(f'[mmPoint] {ckpt}')
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
    if hasattr(trainer.optimizer,'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    trainer.model.eval(); trainer.compressor.eval()
    ae_cfg = getattr(cfg,'geo_ae',None)
    ae = GeoAE(cfg=ae_cfg,
               feat_dim=getattr(ae_cfg,'feat_dim',256) if ae_cfg else 256,
               num_coarse=getattr(ae_cfg,'num_coarse',128) if ae_cfg else 128).to(device)
    if ae_ckpt and os.path.exists(ae_ckpt):
        ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
    ae.eval(); print(f'[PRISM]  epoch={ck.get("epoch","?")}')
    return trainer, ae, cfg


# ══════════════════════════════════════════════════════════════════════════════
#  推理
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
    return preds[-1][0].cpu().numpy()   # finest output (B,2048,3)

@torch.no_grad()
def infer_mmpoint(model, template, mm512, device):
    outs = model(torch.from_numpy(mm512).float().unsqueeze(0).to(device), template)
    return outs[2][0].cpu().numpy()   # deform3: finest output (2048, 3)

@torch.no_grad()
def infer_prism(trainer, ae, fd, action_map, action, cfg, device):
    mm   = torch.from_numpy(fd['mm_vis256']).float().unsqueeze(0).to(device)
    skel = torch.from_numpy(fd['skel_norm']).unsqueeze(0).to(device)
    lbl  = torch.tensor([action_map.get(action,0)],dtype=torch.long).to(device)
    coarse,_ = ae(mm)
    if hasattr(trainer.optimizer,'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    zs = (cfg.score.z_scale,
          cfg.score.z_dim+3 if cfg.score.graphconv else cfg.score.z_dim)
    z_list = trainer.SDE.sample_discrete(
        score_fn=trainer.score_fn, num_samples=1, N=cfg.sde.sample_N,
        corrector=cfg.sde.corrector, predictor=cfg.sde.predictor,
        corrector_steps=cfg.sde.corrector_steps, shape=zs,
        time_eps=trainer.sample_time_eps, label=lbl,
        denoise=cfg.sde.denoise, device=device,
        probability_flow=cfg.sde.probability_flow, snr=cfg.sde.snr,
        condition={'skeleton':skel,'coarse':coarse},
        print_steps=cfg.sde.sample_N)
    pc = trainer.compressor.sample((1,cfg.common.num_points), given_eps=z_list[-1])
    if hasattr(trainer.optimizer,'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    return pc[0].cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  逐格渲染（render each cell → numpy RGBA array）
# ══════════════════════════════════════════════════════════════════════════════

CELL_PX  = 320   # pixels per cell (square)
CELL_DPI = 150
CELL_IN  = CELL_PX / CELL_DPI

# ── per-method point size / alpha ────────────────────────────────────────────
_RENDER_STYLE = {
    'mmwave':    dict(s=28, alpha=0.92, depthshade=False),
    'lidar':     dict(s=5,  alpha=0.95, depthshade=False),
    'prism':     dict(s=5,  alpha=0.97, depthshade=False),
    'pcn':       dict(s=3,  alpha=0.90, depthshade=False),
    'pointr':    dict(s=3,  alpha=0.90, depthshade=False),
    'snowflake': dict(s=3,  alpha=0.90, depthshade=False),
    'mmpoint':   dict(s=3,  alpha=0.90, depthshade=False),
}


def _cell_img(pc, ref_pc, row_key, vmin, vmax):
    """Render one point cloud cell coloured by Z-height (viridis).
    vmin/vmax are shared across all cells for a consistent colour scale."""
    fig = plt.figure(figsize=(CELL_IN, CELL_IN), facecolor='white', dpi=CELL_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    fig.patch.set_alpha(1.)

    # ── coordinate bounds from reference LiDAR GT (consistent across all cells) ──
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
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)

    style = _RENDER_STYLE.get(row_key, dict(s=3, alpha=0.90, depthshade=False))
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],
               c=pc[:,2], cmap='viridis', vmin=vmin, vmax=vmax,
               s=style['s'], alpha=style['alpha'],
               depthshade=style['depthshade'], linewidths=0)

    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf, dtype=np.uint8).reshape(CELL_PX, CELL_PX, 4).copy()
    plt.close(fig)
    return img   # RGBA uint8


def _white_bg(rgba):
    """Composite RGBA onto white, return RGB uint8."""
    rgb = rgba[:,:,:3].astype(np.float32) / 255.
    a   = rgba[:,:,3:4].astype(np.float32) / 255.
    out = rgb * a + (1. - a)
    return (out * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  최종 조립  (assemble grid + labels with matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

ROWS       = ['pcn', 'pointr', 'snowflake', 'mmpoint', 'prism', 'lidar']
ROW_LABELS = ['PCN$^*$', 'PoinTr$^*$', 'SnowflakeNet$^*$', 'mmPoint$^*$', 'PRISM\n(Ours)', 'LiDAR GT']
ROW_BOLD   = [False, False, False, False, True, False]


def _autocrop(img, pad=6):
    """Trim white border and re-add a small uniform margin."""
    mask = ~((img[:,:,0] > 250) & (img[:,:,1] > 250) & (img[:,:,2] > 250))
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0 = max(0, rows[0] - pad);  r1 = min(img.shape[0], rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad);  c1 = min(img.shape[1], cols[-1] + pad + 1)
    return img[r0:r1, c0:c1]

def _uniform_crop(cell_imgs):
    """Autocrop every cell then pad to a common size (max content bbox)."""
    cropped = {a: {} for a in ACTIONS}
    max_h, max_w = 0, 0
    for action in ACTIONS:
        for rk in ROWS:
            c = _autocrop(cell_imgs[action][rk])
            cropped[action][rk] = c
            max_h = max(max_h, c.shape[0])
            max_w = max(max_w, c.shape[1])
    for action in ACTIONS:
        for rk in ROWS:
            img = cropped[action][rk]
            h, w = img.shape[:2]
            canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            y0 = (max_h - h) // 2;  x0 = (max_w - w) // 2
            canvas[y0:y0+h, x0:x0+w] = img
            cropped[action][rk] = canvas
    return cropped, max_h, max_w

def assemble_and_save(cell_imgs, out_path, dpi):
    """
    Layout (wide figure for NeurIPS single-column):
      rows = actions  (ACTIONS)
      cols = methods  (ROWS)
    Left margin : action names  (row labels)
    Top  margin : method names  (col labels)
    """
    cell_imgs, cell_h, cell_w = _uniform_crop(cell_imgs)

    # ── Build pixel grid: rows=actions, cols=methods ──
    grid_rows = []
    for action in ACTIONS:
        strip = np.hstack([cell_imgs[action][rk] for rk in ROWS])
        grid_rows.append(strip)
    grid = np.vstack(grid_rows)

    grid_h_px, grid_w_px = grid.shape[:2]
    grid_h_in = grid_h_px / dpi
    grid_w_in = grid_w_px / dpi

    # Margins (inches) for labels
    margin_left = 0.52   # action name labels
    margin_top  = 0.28   # method name labels

    fig_w = margin_left + grid_w_in
    fig_h = margin_top  + grid_h_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='white')

    # ── Main image axes ────────────────────────────────────────────────────
    ax = fig.add_axes([margin_left / fig_w, 0.0,
                       grid_w_in   / fig_w, grid_h_in / fig_h])
    ax.imshow(grid, aspect='equal', interpolation='lanczos')
    ax.set_axis_off()

    # ── Column labels (method names, at top) ──────────────────────────────
    for j, (label, bold) in enumerate(zip(ROW_LABELS, ROW_BOLD)):
        x = (margin_left + (j + 0.5) * cell_w / dpi) / fig_w
        y = (grid_h_in + margin_top * 0.55) / fig_h
        color  = '#c0392b' if bold else '#111111'
        weight = 'bold'    if bold else 'semibold'
        fig.text(x, y, label.replace('\n', ' '),
                 ha='center', va='center',
                 fontsize=6.5, fontweight=weight, color=color,
                 transform=fig.transFigure)

    # ── Row labels (action names, at left, rotated) ────────────────────────
    for i, action in enumerate(ACTIONS):
        # row i starts at top of grid in figure coords
        row_center_y = (grid_h_in - (i + 0.5) * cell_h / dpi) / fig_h
        x = margin_left * 0.45 / fig_w
        fig.text(x, row_center_y, ACTION_NAMES[action],
                 ha='center', va='center',
                 fontsize=6.5, fontweight='normal', color='#111111',
                 rotation=90, transform=fig.transFigure)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor='white',
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'\n✓  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcn_ckpt',
        default='baseline/PCN/output_mmfi/PCN-0319_1438/checkpoints/bestl1_network.pth')
    parser.add_argument('--pointr_ckpt',
        default='baseline/PoinTr/output_mmfi/PoinTr-0319_1503/checkpoints/bestl1_network.pth')
    parser.add_argument('--snowflake_ckpt',
        default='baseline/SnowflakeNet/output_mmfi/SnowflakeNet-0327_0113/checkpoints/bestl1_network.pth')
    parser.add_argument('--prism_ckpt',
        default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_3600.pth')
    parser.add_argument('--ae_ckpt',
        default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/autoencoder_e_3600.pth')
    parser.add_argument('--mmpoint_ckpt',
        default='baseline/mmPoint/output_mmfi/mmPoint_mmfi-0319_1339/checkpoints/bestl1_network.pth')
    parser.add_argument('--cfg',  default=CFG_PATH)
    parser.add_argument('--out',  default='assets/qual_comparison.png')
    parser.add_argument('--dpi',  type=int, default=200)
    parser.add_argument('--gpu',  type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\n[1] Loading models...')
    pcn       = load_pcn(args.pcn_ckpt, device)
    pointr    = load_pointr(args.pointr_ckpt, device)
    snowflake = load_snowflake(args.snowflake_ckpt, device)
    mmpoint, mm_tmpl = load_mmpoint(args.mmpoint_ckpt, device)
    trainer, ae, cfg = load_prism(args.cfg, args.prism_ckpt, args.ae_ckpt, device)
    # NOTE: PRISM score network was trained with 8 actions only.
    # Keep the original 8-action mapping to avoid embedding index OOB.
    action_map = {a:i for i,a in enumerate(sorted(
        ['A03','A12','A13','A17','A19','A22','A26','A27']))}

    print('\n[2] Loading frames...')
    frames = {a: load_frame(a) for a in ACTIONS}

    print('\n[3] Running inference...')
    raw = {}
    for action in ACTIONS:
        fd = frames[action]
        print(f'  {action}: PCN', end='', flush=True)
        pc_pcn = infer_pcn(pcn, fd['mm_vis512'], device)
        print(' | PoinTr', end='', flush=True)
        pc_ptr = infer_pointr(pointr, fd['mm_vis512'], device)
        print(' | SnowflakeNet', end='', flush=True)
        pc_sfn = infer_snowflake(snowflake, fd['mm_vis512'], device)
        print(' | mmPoint', end='', flush=True)
        pc_mmp = infer_mmpoint(mmpoint, mm_tmpl, fd['mm_vis512'], device)
        print(' | PRISM (full SDE)...', flush=True)
        pc_pr  = infer_prism(trainer, ae, fd, action_map, action, cfg, device)
        raw[action] = dict(
            mmwave    = fd['mm_vis256'],
            pcn       = _norm256(pc_pcn),
            pointr    = _norm256(pc_ptr),
            snowflake = _norm256(pc_sfn),
            mmpoint   = _norm256(pc_mmp),
            prism     = pc_pr,
            lidar     = fd['lidar_vis'],
        )

    print('\n[4] Rendering cells...')
    # ── global Z range from all LiDAR GT frames (shared colour scale) ──
    all_z = np.concatenate([raw[a]['lidar'][:, 2] for a in ACTIONS])
    vmin, vmax = float(all_z.min()), float(all_z.max())

    cell_imgs = {a: {} for a in ACTIONS}
    for r, row_key in enumerate(ROWS):
        for c, action in enumerate(ACTIONS):
            ref = raw[action]['lidar']
            rgba = _cell_img(raw[action][row_key], ref, row_key, vmin, vmax)
            cell_imgs[action][row_key] = _white_bg(rgba)
            sys.stdout.write(f'\r  {r*len(ACTIONS)+c+1}/{len(ROWS)*len(ACTIONS)}')
            sys.stdout.flush()
    print()

    print('[5] Assembling final figure...')
    assemble_and_save(cell_imgs, args.out, args.dpi)
    print(f'    Open with: eog {args.out}')


if __name__ == '__main__':
    main()
