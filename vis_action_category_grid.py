"""
vis_action_category_grid.py
===========================
绘制 8 个动作的最终重建点云（2×4 网格）：
  行 0 — 康复动作 (Rehabilitation)：A03 A12 A13 A22   蓝色
  行 1 — 日常动作 (Daily Living)   ：A17 A19 A26 A27   橙红色

用法：
  python vis_action_category_grid.py \
      --ckpt   experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_500.pth \
      --ae_ckpt experiments/Latent_Diffusion_Trainer/mmfi_202603151358/geo_ae_500.pth \
      --out assets/fig/action_category_grid.png \
      --dpi 200
"""

import argparse, os, sys, glob, re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.scorenet.score import Score
from model.Compressor.Network import Compressor
from model.GeoAE.Network import GeoAE
from completion_trainer.Latent_SDE_Trainer import Trainer
from tools.io import dict2namespace
import yaml

# ── 固定配置 ────────────────────────────────────────────────────
DATA_ROOT    = 'data/MMFi/E01'
CFG_PATH     = 'experiments/Latent_Diffusion_Trainer/config_8actions.yaml'
MMWAVE_N     = 256
SUBJECT      = 'S01'

# 康复动作（蓝色）
REHAB_ACTIONS = ['A03', 'A12', 'A13', 'A22']
REHAB_FRAMES  = [  29,    8,    73,    50  ]
REHAB_COLOR   = '#4D96FF'

# 日常动作（橙红色）
DAILY_ACTIONS = ['A17', 'A19', 'A26', 'A27']
DAILY_FRAMES  = [  44,    59,    40,    51  ]
DAILY_COLOR   = '#FF6B35'

ACTION_NAMES = {
    'A03': 'Chest V',    'A12': 'Squat',
    'A13': 'Raise Hand', 'A22': 'Kick',
    'A17': 'Wave Hand',  'A19': 'Pick Up',
    'A26': 'Jump Up',    'A27': 'Bowing',
}

TRAIN_ACTIONS = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']


# ═══════════════════════════════════════════════════════════════
#  数据 I/O（复用 vis_diffusion_grid.py）
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

def build_action_map(cfg):
    return {act: i for i, act in enumerate(sorted(cfg.data.actions))}

def load_one_frame(subject, action, action_map, frame_idx):
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
    print(f'    {subject}/{action} 帧 #{fn}')

    lidar = _load_lidar(l_files[fn])
    lidar_norm, shift, scale = _normalize(lidar)

    mm_xyz = _load_mmwave(m_files[fn])
    valid  = np.isfinite(mm_xyz).all(1) & (np.abs(mm_xyz)<20.).all(1)
    mm_xyz = mm_xyz[valid]
    if mm_xyz.shape[0] == 0: mm_xyz = np.zeros((1,3), dtype=np.float32)
    mm_norm, _, _ = _normalize(mm_xyz)
    mm_vis    = _subsample(mm_norm, MMWAVE_N)
    mm_tensor = torch.from_numpy(mm_vis).float().unsqueeze(0)

    skel_raw    = gt[fn-1][:, [2,0,1]].copy()
    skel_norm   = (skel_raw - shift) / scale
    skel_tensor = torch.from_numpy(skel_norm.astype(np.float32)).unsqueeze(0)

    return {
        'mm_tensor':   mm_tensor,
        'skel_tensor': skel_tensor,
        'label':       action_map.get(action, 0),
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
    trainer.epoch = ckpt.get('epoch', 0) + 1
    trainer.itr   = ckpt.get('itr', 0)
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
#  推理：只取最终重建结果
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_final(trainer, ae, frame_data, device):
    mm    = frame_data['mm_tensor'].to(device)
    skel  = frame_data['skel_tensor'].to(device)
    label = torch.tensor([frame_data['label']], dtype=torch.long).to(device)
    coarse, _ = ae(mm)
    pts, _ = trainer.sample(1, label=label,
                            condition={'skeleton': skel, 'coarse': coarse})
    return pts[0].cpu().numpy()


# ═══════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════

def style_3d_nobox(ax):
    """去掉坐标框，只显示点云。"""
    ax.set_axis_off()
    ax.set_facecolor('white')


def render_grid(rehab_pcs, daily_pcs, out_path,
                elev=20, azim=-60, dpi=150, point_size=4, extra_blue=()):
    N_COLS = 4   # 每行 4 个动作

    cell_w, cell_h = 2.2, 2.6
    left_pad = 1.1   # 行类别标签宽度
    top_pad  = 0.55  # 顶部列标题高度
    bot_pad  = 0.15
    fig_w = left_pad + N_COLS * cell_w
    fig_h = top_pad + 2 * cell_h + bot_pad

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    gs = gridspec.GridSpec(
        2, N_COLS, figure=fig,
        left   = left_pad / fig_w,
        right  = 0.995,
        top    = 1.0 - top_pad / fig_h,
        bottom = bot_pad / fig_h,
        wspace = 0.02,
        hspace = 0.05,
    )

    rows = [
        (0, REHAB_ACTIONS, rehab_pcs, REHAB_COLOR, 'Rehabilitation'),
        (1, DAILY_ACTIONS, daily_pcs, DAILY_COLOR,  'Daily Living'),
    ]

    for row_i, actions, pcs, color, category in rows:
        for col_i, (action, pc) in enumerate(zip(actions, pcs)):
            ax = fig.add_subplot(gs[row_i, col_i], projection='3d')
            style_3d_nobox(ax)
            ax.view_init(elev=elev, azim=azim)

            # 单独动作颜色覆盖
            c = REHAB_COLOR if action in extra_blue else color

            # 固定等比例坐标范围
            mn, mx = pc.min(0), pc.max(0)
            mid = (mn + mx) / 2
            r   = max((mx - mn).max() / 2 * 1.3, 0.3)
            ax.set_xlim(mid[0]-r, mid[0]+r)
            ax.set_ylim(mid[1]-r, mid[1]+r)
            ax.set_zlim(mid[2]-r, mid[2]+r)

            ax.scatter(pc[:,0], pc[:,1], pc[:,2],
                       c=c, s=point_size, alpha=0.88,
                       depthshade=False, linewidths=0)

            # 动作标题（底部，每格独立）
            aname = ACTION_NAMES.get(action, action)
            ax.set_title(f'{action}\n{aname}',
                         color=c, fontsize=9, fontweight='bold',
                         pad=2, loc='center')

            # 行类别标签（只在每行第一格左侧画一次）
            if col_i == 0:
                ax.text2D(-0.22, 0.5,
                          category,
                          transform=ax.transAxes,
                          color=color, fontsize=10, fontweight='bold',
                          va='center', ha='right', rotation=90,
                          linespacing=1.4)

    # 全图标题
    fig.text(
        (left_pad / fig_w + 0.995) / 2,
        1.0 - top_pad / fig_h / 2,
        'PRISM — Reconstructed Point Clouds by Action Category',
        ha='center', va='center',
        color='#222222', fontsize=11, fontweight='bold',
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor='white',
                bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'  已保存: {out_path}')


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('PRISM 动作类别重建点云网格图')
    parser.add_argument('--cfg',        default=CFG_PATH)
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--ae_ckpt',    required=True)
    parser.add_argument('--subject',    default=SUBJECT)
    parser.add_argument('--rehab_frames', type=int, nargs=4,
                        default=REHAB_FRAMES,
                        metavar=('A03', 'A12', 'A13', 'A22'),
                        help='康复动作各取第几帧 (A03 A12 A13 A22)，默认: %(default)s')
    parser.add_argument('--daily_frames', type=int, nargs=4,
                        default=DAILY_FRAMES,
                        metavar=('A17', 'A19', 'A26', 'A27'),
                        help='日常动作各取第几帧 (A17 A19 A26 A27)，默认: %(default)s')
    parser.add_argument('--extra_blue',  nargs='*', default=[],
                        metavar='ACTION',
                        help='强制显示为蓝色的动作ID，例如 --extra_blue A19 A27')
    parser.add_argument('--out',        default='assets/fig/action_category_grid.png')
    parser.add_argument('--elev',       type=float, default=20)
    parser.add_argument('--azim',       type=float, default=-60)
    parser.add_argument('--point_size', type=float, default=4)
    parser.add_argument('--dpi',        type=int,   default=150)
    parser.add_argument('--gpu',        type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\n[1] 加载模型...')
    trainer, ae, cfg = load_models(args.cfg, args.ckpt, args.ae_ckpt, device)
    action_map = build_action_map(cfg)

    all_actions = REHAB_ACTIONS + DAILY_ACTIONS
    all_frames  = args.rehab_frames + args.daily_frames

    print('\n[2] 加载各动作代表帧...')
    all_frame_data = []
    for action, fidx in zip(all_actions, all_frames):
        fd = load_one_frame(args.subject, action, action_map, fidx)
        all_frame_data.append(fd)

    print('\n[3] 模型推理（每动作约 30-60 秒）...')
    all_pcs = []
    for i, (action, fd) in enumerate(zip(all_actions, all_frame_data)):
        print(f'  [{i+1}/8] {action} {ACTION_NAMES.get(action,"")}')
        pc = sample_final(trainer, ae, fd, device)
        all_pcs.append(pc)

    rehab_pcs = all_pcs[:4]
    daily_pcs = all_pcs[4:]

    print('\n[4] 渲染合成图...')
    render_grid(rehab_pcs, daily_pcs, args.out,
                args.elev, args.azim, args.dpi, args.point_size,
                extra_blue=args.extra_blue)

    print(f'\n完成！图像保存至: {args.out}')


if __name__ == '__main__':
    main()
