"""
vis_diffusion_web.py
====================
为网页互动展示生成逐帧扩散去噪图像。

对每个动作分别运行逆 SDE，将各时间步保存为独立 PNG：
  assets/diffusion/<action>/step_NN.png   (N=00: 纯噪声, N=n_steps-1: 最终生成)
  assets/diffusion/<action>/input.png     (mmWave 雷达输入)
  assets/diffusion/<action>/gt.png        (LiDAR 参考)

同时保存：
  assets/diffusion/manifest.json

用法：
  python vis_diffusion_web.py \\
      --ckpt   experiments/Latent_Diffusion_Trainer/.../checkpt_best.pth \\
      --ae_ckpt experiments/Latent_Diffusion_Trainer/.../geo_ae_500.pth \\
      --n_steps 12
"""

import argparse, os, sys, glob, re, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.scorenet.score import Score
from model.Compressor.Network import Compressor
from model.GeoAE.Network import GeoAE
from completion_trainer.Latent_SDE_Trainer import Trainer
from tools.io import dict2namespace
import yaml

# ── 固定配置 ────────────────────────────────────────────────
DATA_ROOT  = 'data/MMFi/E01'
CFG_PATH   = 'experiments/Latent_Diffusion_Trainer/config.yaml'
MMWAVE_N   = 256
SUBJECT    = 'S01'

ACTIONS   = ['A03', 'A12', 'A13', 'A19', 'A22', 'A26', 'A27', 'A17']
FRAME_IDX = [  29,    8,    73,    59,    50,    40,    51,    44  ]

ACTION_COLORS = {
    'A03': '#4D96FF', 'A12': '#FF6B6B', 'A13': '#48BB78', 'A17': '#F72585',
    'A19': '#FF9A3C', 'A22': '#9B59B6', 'A26': '#D4A017', 'A27': '#00B4D8',
}
ACTION_NAMES = {
    'A03': 'Chest V',   'A12': 'Squat',      'A13': 'Raise Hand',
    'A17': 'Wave Hand', 'A19': 'Pick Up',     'A22': 'Kick',
    'A26': 'Jump Up',   'A27': 'Bowing',
}


# ═══════════════════════════════════════════════════════════════
#  数据 I/O（与 vis_diffusion_grid.py 相同）
# ═══════════════════════════════════════════════════════════════

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0: return np.zeros((1,3), dtype=np.float32)
    return raw[:(raw.size//3)*3].reshape(-1,3).astype(np.float32)

_MM_MAGIC   = bytes([2,1,4,3,6,5,8,7])
_MM_HDR_FMT = "<QIIIIIIII"
_MM_HDR_SIZE = 40

def _try_numpy_float32_mm(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0: return None
    if raw.size % 5 == 0:
        pc = raw.reshape(-1,5)
        if np.all(np.abs(pc[:,:3]) < 20): return pc
    if raw.size % 4 == 0:
        pc = raw.reshape(-1,4)
        if np.all(np.abs(pc[:,:3]) < 20):
            return np.hstack([pc, np.zeros((pc.shape[0],1), dtype=np.float32)])
    return None

def _parse_ti_uart(path):
    import struct as _struct
    raw = open(path,'rb').read()
    all_pts, pos = [], 0
    while True:
        idx = raw.find(_MM_MAGIC, pos)
        if idx == -1: break
        if idx+_MM_HDR_SIZE > len(raw): break
        total_len = _struct.unpack_from('<I', raw, idx+12)[0]
        frame = raw[idx: idx+total_len]
        if len(frame) < _MM_HDR_SIZE: pos = idx+_MM_HDR_SIZE; continue
        try: header = _struct.unpack_from(_MM_HDR_FMT, frame, 0)
        except _struct.error: pos = idx+_MM_HDR_SIZE; continue
        if header[0] != _struct.unpack('<Q', _MM_MAGIC)[0]: pos = idx+_MM_HDR_SIZE; continue
        pts, offset = [], _MM_HDR_SIZE
        for _ in range(header[7]):
            if offset+8 > len(frame): break
            tlv_type, tlv_len = _struct.unpack_from('<II', frame, offset); offset += 8
            if tlv_type == 1:
                for _ in range(tlv_len//16):
                    if offset+16 > len(frame): break
                    x,y,z,v = _struct.unpack_from('<ffff', frame, offset)
                    pts.append([x,y,z,v,0.0]); offset += 16
            elif tlv_type == 7:
                for _ in range(tlv_len//20):
                    if offset+20 > len(frame): break
                    x,y,z,v = _struct.unpack_from('<ffff', frame, offset)
                    snr,_ = _struct.unpack_from('<HH', frame, offset+16)
                    pts.append([x,y,z,v,float(snr)]); offset += 20
            else: offset += tlv_len
        if pts: all_pts.append(np.array(pts, dtype=np.float32))
        pos = max(idx+_MM_HDR_SIZE, idx+total_len)
    return np.vstack(all_pts) if all_pts else None

def _load_mmwave(path):
    pc = _try_numpy_float32_mm(path)
    if pc is not None and len(pc)>0: return pc[:,:3]
    pc = _parse_ti_uart(path)
    if pc is not None and len(pc)>0: return pc[:,:3]
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size % 5 == 0 and raw64.size > 0:
        return raw64.reshape(-1,5).astype(np.float32)[:,:3]
    return np.zeros((1,3), dtype=np.float32)

def _normalize(pc):
    c = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s = max(np.max(np.sqrt((pc_c**2).sum(1))), 1e-8)
    return (pc_c/s).astype(np.float32), c.astype(np.float32), float(s)

def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0: return np.zeros((n,3), dtype=np.float32)
    return pc[np.random.choice(N, n, replace=N<n)]


# ═══════════════════════════════════════════════════════════════
#  加载单帧
# ═══════════════════════════════════════════════════════════════

def build_action_map(cfg):
    return {act: i for i, act in enumerate(sorted(cfg.data.actions))}

def load_one_frame(subject, action, action_map, frame_idx=30):
    lidar_dir  = os.path.join(DATA_ROOT, subject, action, 'lidar')
    mmwave_dir = os.path.join(DATA_ROOT, subject, action, 'mmwave')
    gt_path    = os.path.join(DATA_ROOT, subject, action, 'ground_truth.npy')

    gt_raw = np.load(gt_path, allow_pickle=True)
    if gt_raw.dtype == object: gt_raw = np.array(gt_raw.tolist(), dtype=np.float32)
    else: gt_raw = gt_raw.astype(np.float32)
    s = gt_raw.shape
    if   gt_raw.ndim==3 and s[1]==17 and s[2]==3: gt = gt_raw
    elif gt_raw.ndim==3 and s[1]==3  and s[2]==17: gt = gt_raw.transpose(0,2,1)
    elif gt_raw.ndim==2 and s[1]==51: gt = gt_raw.reshape(s[0],17,3)
    else: raise ValueError(f'无法解析 GT: {gt_raw.shape}')

    l_files = {_frame_num(f): f for f in glob.glob(os.path.join(lidar_dir, 'frame*_filtered.bin'))}
    m_files = {_frame_num(f): f for f in glob.glob(os.path.join(mmwave_dir,'frame*_filtered.bin'))}
    l_files.pop(-1,None); m_files.pop(-1,None)
    common = sorted(set(l_files) & set(m_files))
    common = [fn for fn in common if fn-1 < len(gt)]

    fn = common[min(frame_idx, len(common)-1)]
    print(f'    帧 #{fn}')

    lidar = _load_lidar(l_files[fn])
    lidar_norm, shift, scale = _normalize(lidar)
    lidar_vis = _subsample(lidar_norm, 256)

    mm_xyz = _load_mmwave(m_files[fn])
    valid  = np.isfinite(mm_xyz).all(1) & (np.abs(mm_xyz)<20.).all(1)
    mm_xyz = mm_xyz[valid]
    if mm_xyz.shape[0] == 0: mm_xyz = np.zeros((1,3), dtype=np.float32)
    mm_norm, _, _ = _normalize(mm_xyz)
    mm_vis    = _subsample(mm_norm, MMWAVE_N)
    mm_tensor = torch.from_numpy(mm_vis).float().unsqueeze(0)

    skel_raw  = gt[fn-1][:, [2,0,1]].copy()
    skel_norm = (skel_raw - shift) / scale
    skel_tensor = torch.from_numpy(skel_norm.astype(np.float32)).unsqueeze(0)

    label = action_map.get(action, 0)
    return {
        'mm_tensor':   mm_tensor,
        'skel_tensor': skel_tensor,
        'lidar_vis':   lidar_vis,
        'mmwave_vis':  mm_vis,
        'label':       label,
        'shift':       shift,
        'scale':       scale,
    }


# ═══════════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════════

def load_models(cfg_path, ckpt_path, ae_ckpt_path, device):
    with open(cfg_path) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader))
    model      = Score(cfg.score)
    compressor = Compressor(cfg.compressor)
    trainer    = Trainer(cfg, model=model, compressor=compressor, device=device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    trainer.model.load_state_dict(ckpt['score_state_dict'], strict=True)
    trainer.compressor.load_state_dict(ckpt['compressor_state_dict'], strict=True)
    trainer.compressor.init()
    trainer.epoch = ckpt.get('epoch',0) + 1
    trainer.itr   = ckpt.get('itr',0)
    if hasattr(trainer.optimizer, 'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    trainer.model.eval(); trainer.compressor.eval()
    print(f'  LDT loaded epoch={ckpt.get("epoch","?")}')

    ae_cfg = getattr(cfg, 'geo_ae', None)
    ae = GeoAE(
        cfg=ae_cfg,
        feat_dim=getattr(ae_cfg,'feat_dim',256) if ae_cfg else 256,
        num_coarse=getattr(ae_cfg,'num_coarse',128) if ae_cfg else 128,
    ).to(device)
    if ae_ckpt_path and os.path.exists(ae_ckpt_path):
        ae.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
    ae.eval()
    return trainer, ae, cfg


# ═══════════════════════════════════════════════════════════════
#  带中间步输出的采样
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_with_steps(trainer, ae, frame_data, n_steps, device):
    mm    = frame_data['mm_tensor'].to(device)
    skel  = frame_data['skel_tensor'].to(device)
    label = torch.tensor([frame_data['label']], dtype=torch.long).to(device)

    coarse, _ = ae(mm)
    condition  = {'skeleton': skel, 'coarse': coarse}

    trainer.model.eval(); trainer.compressor.eval()
    trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    cfg     = trainer.cfg
    z_shape = (cfg.score.z_scale,
               cfg.score.z_dim + 3 if cfg.score.graphconv else cfg.score.z_dim)

    z_list = trainer.SDE.sample_discrete(
        score_fn=trainer.score_fn,
        num_samples=1,
        N=cfg.sde.sample_N,
        corrector=cfg.sde.corrector,
        predictor=cfg.sde.predictor,
        corrector_steps=cfg.sde.corrector_steps,
        shape=z_shape,
        time_eps=trainer.sample_time_eps,
        label=label,
        denoise=cfg.sde.denoise,
        device=device,
        probability_flow=cfg.sde.probability_flow,
        snr=cfg.sde.snr,
        condition=condition,
        print_steps=n_steps,
    )

    num_pts = cfg.common.num_points
    pc_list = []
    for z in z_list:
        pc = trainer.compressor.sample((1, num_pts), given_eps=z)
        pc_list.append(pc[0].cpu().numpy())

    trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    return pc_list


# ═══════════════════════════════════════════════════════════════
#  渲染单帧
# ═══════════════════════════════════════════════════════════════

BG = '#F8F8F8'

def _setup_ax(ax, elev, azim, mid, r):
    ax.set_facecolor(BG)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#E8E8E8')
    ax.tick_params(colors='#CCCCCC', labelsize=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


def render_step(pc, step_idx, n_steps, ref_pc, out_path,
                rng, color='#4D96FF', elev=20, azim=-60, dpi=110):
    """渲染去噪过程中的某一步（噪声+解码点云混合）。"""
    t       = np.linspace(1.0, 0.0, n_steps)[step_idx]
    mn, mx  = ref_pc.min(0), ref_pc.max(0)
    mid     = (mn+mx)/2
    r       = max((mx-mn).max()/2 * 1.35, 0.3)

    fig = plt.figure(figsize=(4.2, 4.2), facecolor=BG)
    ax  = fig.add_subplot(111, projection='3d')
    _setup_ax(ax, elev, azim, mid, r)

    n_pts   = pc.shape[0]
    n_noise = int(n_pts * t * 3)
    n_clean = max(1, int(n_pts * (1 - t * 0.85)))

    if n_clean > 0:
        idx_c   = rng.choice(n_pts, min(n_clean, n_pts), replace=False)
        pc_c    = pc[idx_c]
        alpha_c = float(np.clip(1.0 - t * 0.85, 0.1, 1.0))
        s_size  = float(np.clip(6.5 - t * 5.0, 1.5, 6.5))
        ax.scatter(pc_c[:,0], pc_c[:,1], pc_c[:,2],
                   c=color, s=s_size, alpha=alpha_c,
                   depthshade=False, linewidths=0)

    if n_noise > 0:
        # 均匀分布填满可视框，避免 Gaussian clip 堆积成可见边界
        noise_pts = rng.uniform(-1, 1, size=(n_noise, 3)) * r * 0.96 + mid
        alpha_n   = float(np.clip(t * 0.75, 0.0, 0.65))
        ax.scatter(noise_pts[:,0], noise_pts[:,1], noise_pts[:,2],
                   c=color, s=1.2, alpha=alpha_n,
                   depthshade=False, linewidths=0)

    plt.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=dpi, facecolor=BG,
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


def render_input(mmwave_vis, ref_pc, out_path, elev=20, azim=-60, dpi=110):
    """渲染 mmWave 雷达输入。"""
    mn, mx = ref_pc.min(0), ref_pc.max(0)
    mid    = (mn+mx)/2
    r      = max((mx-mn).max()/2 * 1.35, 0.3)

    fig = plt.figure(figsize=(4.2, 4.2), facecolor=BG)
    ax  = fig.add_subplot(111, projection='3d')
    _setup_ax(ax, elev, azim, mid, r)
    ax.scatter(mmwave_vis[:,0], mmwave_vis[:,1], mmwave_vis[:,2],
               c='#FF7700', s=20, alpha=0.9,
               edgecolors='#FFAA44', linewidths=0.3,
               depthshade=False)
    plt.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=dpi, facecolor=BG,
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


def render_gt(lidar_vis, ref_pc, out_path, elev=20, azim=-60, dpi=110):
    """渲染 LiDAR Ground Truth。"""
    mn, mx  = ref_pc.min(0), ref_pc.max(0)
    mid     = (mn+mx)/2
    r       = max((mx-mn).max()/2 * 1.35, 0.3)
    zlim_lo = ref_pc[:,2].min()
    zlim_hi = ref_pc[:,2].max()

    fig = plt.figure(figsize=(4.2, 4.2), facecolor=BG)
    ax  = fig.add_subplot(111, projection='3d')
    _setup_ax(ax, elev, azim, mid, r)
    ax.scatter(lidar_vis[:,0], lidar_vis[:,1], lidar_vis[:,2],
               c=lidar_vis[:,2], cmap='plasma',
               vmin=zlim_lo, vmax=zlim_hi,
               s=2.5, alpha=0.95, depthshade=False, linewidths=0)
    plt.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=dpi, facecolor=BG,
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('网页扩散过程帧导出')
    parser.add_argument('--ckpt',     required=True)
    parser.add_argument('--ae_ckpt',  required=True)
    parser.add_argument('--cfg',      default=CFG_PATH)
    parser.add_argument('--n_steps',  type=int,   default=0,
                        help='抽取的快照帧数（0=自动使用 config.sde.sample_N）')
    parser.add_argument('--elev',     type=float, default=20)
    parser.add_argument('--azim',     type=float, default=-60)
    parser.add_argument('--dpi',      type=int,   default=110)
    parser.add_argument('--gpu',      type=int,   default=0)
    parser.add_argument('--out_dir',  default='assets/diffusion')
    parser.add_argument('--actions',  nargs='+',  default=ACTIONS,
                        help='要导出的动作列表（默认全部 8 个）')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\n[1] 加载模型...')
    trainer, ae, cfg = load_models(args.cfg, args.ckpt, args.ae_ckpt, device)
    action_map = build_action_map(cfg)

    # 自动使用 config 推理步数（sample_N）作为快照帧数
    n_steps = args.n_steps if args.n_steps > 0 else cfg.sde.sample_N
    print(f'  快照帧数: {n_steps}  (config sample_N={cfg.sde.sample_N})')

    t_values = np.linspace(1.0, 0.0, n_steps).tolist()

    manifest = {
        'n_steps': n_steps,
        't_values': [round(t, 4) for t in t_values],
        'actions': [],
    }

    rng = np.random.default_rng(42)

    for action, frame_idx in zip(ACTIONS, FRAME_IDX):
        if action not in args.actions:
            continue
        print(f'\n[{action}] {ACTION_NAMES[action]}  (frame_idx={frame_idx})')

        out_action_dir = os.path.join(args.out_dir, action)
        os.makedirs(out_action_dir, exist_ok=True)

        print('  → 加载数据帧...')
        frame_data = load_one_frame(SUBJECT, action, action_map, frame_idx)

        print(f'  → 采样 {n_steps} 个时间步...')
        pc_list = sample_with_steps(trainer, ae, frame_data, n_steps, device)

        actual_n = len(pc_list)   # sample_discrete 可能返回 n_steps+1 帧
        ref_pc   = pc_list[-1]    # 最终生成帧用作坐标参考

        # 更新 manifest 的帧数（以第一个动作为准）
        if manifest['n_steps'] != actual_n:
            manifest['n_steps'] = actual_n
            manifest['t_values'] = [round(v, 4)
                                    for v in np.linspace(1.0, 0.0, actual_n).tolist()]

        action_color = ACTION_COLORS[action]
        print('  → 渲染去噪步...')
        for i, pc in enumerate(pc_list):
            out_path = os.path.join(out_action_dir, f'step_{i:03d}.png')
            render_step(pc, i, actual_n, ref_pc, out_path,
                        rng, color=action_color,
                        elev=args.elev, azim=args.azim, dpi=args.dpi)
            sys.stdout.write(f'\r     step {i+1}/{actual_n}')
            sys.stdout.flush()
        print()

        print('  → 渲染 mmWave 输入...')
        render_input(frame_data['mmwave_vis'], ref_pc,
                     os.path.join(out_action_dir, 'input.png'),
                     args.elev, args.azim, args.dpi)

        print('  → 渲染 LiDAR GT...')
        render_gt(frame_data['lidar_vis'], ref_pc,
                  os.path.join(out_action_dir, 'gt.png'),
                  args.elev, args.azim, args.dpi)

        manifest['actions'].append({
            'id':    action,
            'name':  ACTION_NAMES[action],
            'color': ACTION_COLORS[action],
        })
        print(f'  ✓ 保存至 {out_action_dir}/')

    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\n✓ manifest 保存至 {manifest_path}')
    print(f'✓ 完成！共导出 {len(manifest["actions"])} 个动作 × {args.n_steps} 步')
    print(f'\n在 index.html 中使用方式：')
    print(f'  python -m http.server 8080  # 然后访问 http://localhost:8080')


if __name__ == '__main__':
    main()
