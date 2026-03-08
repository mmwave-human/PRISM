"""
vis_skelkd_animation.py

生成 SkelKD 预测四格同步动图：
  [GT骨骼] [预测骨骼] [LiDAR点云] [mmWave点云]

用法：
  # 单动作（训练集内）
  python vis_skelkd_animation.py \
      --ckpt experiments/SkelKD/mmfi_skelkd_202603061614/checkpt_2000.pth \
      --save_dir experiments/SkelKD/mmfi_skelkd_202603061614 \
      --action A27 --subject S01 --start 0 --end 80 --fmt mp4

  # 批量生成训练集全部 8 个动作（每个动作 60 帧）
  python vis_skelkd_animation.py \
      --ckpt experiments/SkelKD/mmfi_skelkd_202603061614/checkpt_2000.pth \
      --save_dir experiments/SkelKD/mmfi_skelkd_202603061614 \
      --all_actions --subject S01 --end 60 --fmt mp4

  # 批量生成，多受试者
  python vis_skelkd_animation.py \
      --ckpt experiments/SkelKD/mmfi_skelkd_202603061614/checkpt_2000.pth \
      --save_dir experiments/SkelKD/mmfi_skelkd_202603061614 \
      --all_actions --subject S01 S08 --end 60 --fmt mp4
"""

import argparse
import os
import glob
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.SkelKD.Network import SkelKD
from tools.io import dict2namespace
import yaml

# ── 常量 ──────────────────────────────────────────────────────
DATA_ROOT  = 'data/MMFi/E01'
CFG_PATH   = 'experiments/SkelKD/config.yaml'
MMWAVE_N   = 100

# config.yaml 中训练的 8 个动作
TRAIN_ACTIONS = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']

# MM-Fi 全部 27 个动作的名称
ACTION_NAMES = {
    'A01': 'Stretching',    'A02': 'Chest_H',       'A03': 'Chest_V',
    'A04': 'Twist_L',       'A05': 'Twist_R',        'A06': 'MarkTime',
    'A07': 'LimbExt_L',     'A08': 'LimbExt_R',      'A09': 'Lunge_LF',
    'A10': 'Lunge_RF',      'A11': 'LimbExt_Both',   'A12': 'Squat',
    'A13': 'RaiseHand_L',   'A14': 'RaiseHand_R',    'A15': 'Lunge_LS',
    'A16': 'Lunge_RS',      'A17': 'WaveHand_L',     'A18': 'WaveHand_R',
    'A19': 'PickUp',        'A20': 'Throw_L',         'A21': 'Throw_R',
    'A22': 'Kick_L',        'A23': 'Kick_R',          'A24': 'BodyExt_L',
    'A25': 'BodyExt_R',     'A26': 'JumpUp',          'A27': 'Bowing',
}

# MM-Fi 17关节骨骼连接对
BONES = [
    (0,1),(0,2),(1,3),(2,4),
    (3,5),(4,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

BONE_COLORS = {
    (0,1):'#FF6B6B',(0,2):'#FF6B6B',(1,3):'#FF6B6B',(2,4):'#FF6B6B',
    (3,5):'#FF6B6B',(4,6):'#FF6B6B',(5,6):'#FFD93D',
    (5,7):'#6BCB77',(7,9):'#6BCB77',
    (6,8):'#4D96FF',(8,10):'#4D96FF',
    (5,11):'#FFD93D',(6,12):'#FFD93D',(11,12):'#FFD93D',
    (11,13):'#C77DFF',(13,15):'#C77DFF',
    (12,14):'#FF9A3C',(14,16):'#FF9A3C',
}


# ═══════════════════════════════════════════════════════════════
#  数据加载（与 MMFiViPC.py 完全一致的处理流程）
# ═══════════════════════════════════════════════════════════════

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.zeros((1, 3), dtype=np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)

_MM_MAGIC   = bytes([2, 1, 4, 3, 6, 5, 8, 7])
_MM_HDR_FMT = "<QIIIIIIII"
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
            return np.hstack([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)])
    return None

def _parse_ti_uart(path):
    import struct as _struct
    raw = open(path, 'rb').read()
    all_pts, pos = [], 0
    while True:
        idx = raw.find(_MM_MAGIC, pos)
        if idx == -1: break
        if idx + _MM_HDR_SIZE > len(raw): break
        total_len = _struct.unpack_from('<I', raw, idx + 12)[0]
        frame = raw[idx: idx + total_len]
        if len(frame) < _MM_HDR_SIZE:
            pos = idx + _MM_HDR_SIZE; continue
        try:
            header = _struct.unpack_from(_MM_HDR_FMT, frame, 0)
        except _struct.error:
            pos = idx + _MM_HDR_SIZE; continue
        if header[0] != _struct.unpack('<Q', _MM_MAGIC)[0]:
            pos = idx + _MM_HDR_SIZE; continue
        pts, offset = [], _MM_HDR_SIZE
        for _ in range(header[7]):
            if offset + 8 > len(frame): break
            tlv_type, tlv_len = _struct.unpack_from('<II', frame, offset)
            offset += 8
            if tlv_type == 1:
                for _ in range(tlv_len // 16):
                    if offset + 16 > len(frame): break
                    x, y, z, v = _struct.unpack_from('<ffff', frame, offset)
                    pts.append([x, y, z, v, 0.0]); offset += 16
            elif tlv_type == 7:
                for _ in range(tlv_len // 20):
                    if offset + 20 > len(frame): break
                    x, y, z, v = _struct.unpack_from('<ffff', frame, offset)
                    snr, _ = _struct.unpack_from('<HH', frame, offset + 16)
                    pts.append([x, y, z, v, float(snr)]); offset += 20
            else:
                offset += tlv_len
        if pts:
            all_pts.append(np.array(pts, dtype=np.float32))
        pos = max(idx + _MM_HDR_SIZE, idx + total_len)
    return np.vstack(all_pts) if all_pts else None

def _load_mmwave(path):
    pc = _try_numpy_float32_mm(path)
    if pc is None or len(pc) == 0:
        pc = _parse_ti_uart(path)
    if pc is None or len(pc) == 0:
        raw64 = np.fromfile(path, dtype=np.float64)
        if raw64.size % 5 == 0 and raw64.size > 0:
            pc = raw64.reshape(-1, 5).astype(np.float32)
    if pc is None or len(pc) == 0:
        return np.zeros((1, 3), dtype=np.float32)
    return pc[:, :3]

def _normalize(pc):
    c = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s = max(np.max(np.sqrt((pc_c ** 2).sum(1))), 1e-8)
    return (pc_c / s).astype(np.float32), c.astype(np.float32), float(s)

def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = np.random.choice(N, n, replace=N < n)
    return pc[idx]


def load_sequence(subject, action, start=0, end=60):
    lidar_dir  = os.path.join(DATA_ROOT, subject, action, 'lidar')
    mmwave_dir = os.path.join(DATA_ROOT, subject, action, 'mmwave')
    gt_path    = os.path.join(DATA_ROOT, subject, action, 'ground_truth.npy')

    gt_raw = np.load(gt_path, allow_pickle=True)
    if gt_raw.dtype == object:
        gt_raw = np.array(gt_raw.tolist(), dtype=np.float32)
    else:
        gt_raw = gt_raw.astype(np.float32)
    s = gt_raw.shape
    if gt_raw.ndim == 3 and s[1] == 17 and s[2] == 3:
        gt = gt_raw
    elif gt_raw.ndim == 3 and s[1] == 3 and s[2] == 17:
        gt = gt_raw.transpose(0, 2, 1)
    elif gt_raw.ndim == 2 and s[1] == 51:
        gt = gt_raw.reshape(s[0], 17, 3)
    else:
        raise ValueError(f'无法解析 GT shape: {gt_raw.shape}')

    l_files = {_frame_num(f): f for f in
               glob.glob(os.path.join(lidar_dir,  'frame*_filtered.bin'))}
    m_files = {_frame_num(f): f for f in
               glob.glob(os.path.join(mmwave_dir, 'frame*_filtered.bin'))}
    l_files.pop(-1, None); m_files.pop(-1, None)

    common = sorted(set(l_files) & set(m_files))
    common = [fn for fn in common if fn - 1 < len(gt)]
    common = common[start:end]

    if not common:
        raise RuntimeError(f'没有找到有效帧：{subject}/{action}')

    print(f'  加载 {subject}/{action}，帧 [{common[0]}, {common[-1]}]，共 {len(common)} 帧')

    frames = []
    for fn in common:
        lidar = _load_lidar(l_files[fn])
        lidar_norm, shift, scale = _normalize(lidar)
        lidar_sub  = _subsample(lidar_norm, 512)

        mm_xyz = _load_mmwave(m_files[fn])
        mm_xyz = mm_xyz[:, [1, 0, 2]]
        valid  = (np.isfinite(mm_xyz).all(1) & (np.abs(mm_xyz) < 20.).all(1))
        mm_xyz = mm_xyz[valid]
        if mm_xyz.shape[0] == 0:
            mm_xyz = np.zeros((1, 3), dtype=np.float32)
        mm_xyz = mm_xyz[:, [1, 0, 2]]
        mm_norm, _, _ = _normalize(mm_xyz)
        mm_sub  = _subsample(mm_norm, MMWAVE_N)

        skel_raw    = gt[fn - 1][:, [2, 0, 1]].copy()
        skel_center = skel_raw.mean(axis=0, keepdims=True)
        skel_scale  = max(
            np.max(np.sqrt(((skel_raw - skel_center) ** 2).sum(1))), 1e-8
        )
        skel_norm        = (skel_raw - skel_center) / skel_scale
        skel_norm[:, 2] *= -1

        mm_tensor = torch.from_numpy(mm_sub).float().unsqueeze(0)

        frames.append({
            'lidar':     lidar_sub,
            'mmwave':    mm_sub,
            'mm_tensor': mm_tensor,
            'gt_skel':   skel_norm,
            'shift':     shift,
            'scale':     scale,
            'frame_num': fn,
        })

    return frames


# ═══════════════════════════════════════════════════════════════
#  模型推理
# ═══════════════════════════════════════════════════════════════

def load_model(cfg_path, ckpt_path, device):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2namespace(cfg)
    model = SkelKD(cfg.model).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'  SkelKD loaded from epoch {ckpt.get("epoch", "?")}')
    return model

@torch.no_grad()
def predict_skeletons(model, frames, device):
    preds = []
    for fr in frames:
        mm = fr['mm_tensor'].to(device)
        pred = model(mm).squeeze(0).cpu().numpy()
        preds.append(pred)
    return preds


# ═══════════════════════════════════════════════════════════════
#  可视化工具
# ═══════════════════════════════════════════════════════════════

def draw_skeleton_3d(ax, joints, alpha=1.0, lw=1.8):
    for (a, b) in BONES:
        color = BONE_COLORS.get((a,b), BONE_COLORS.get((b,a), '#AAAAAA'))
        ax.plot([joints[a,0], joints[b,0]],
                [joints[a,1], joints[b,1]],
                [joints[a,2], joints[b,2]],
                color=color, lw=lw, alpha=alpha)
    ax.scatter(joints[:,0], joints[:,1], joints[:,2],
               c='white', s=12, zorder=5, alpha=alpha,
               edgecolors='gray', linewidths=0.5)

def style_ax(ax, title):
    ax.set_facecolor('#0D1117')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#222222')
    ax.yaxis.pane.set_edgecolor('#222222')
    ax.zaxis.pane.set_edgecolor('#222222')
    ax.tick_params(colors='#444444', labelsize=5)
    ax.set_title(title, color='white', fontsize=10, pad=4)
    ax.set_xlabel('X', color='#444444', fontsize=6)
    ax.set_ylabel('Y', color='#444444', fontsize=6)
    ax.set_zlabel('Z', color='#444444', fontsize=6)

def _global_range(arr):
    mn = arr.reshape(-1,3).min(0)
    mx = arr.reshape(-1,3).max(0)
    mid = (mn+mx)/2
    r   = max((mx-mn).max()/2*1.3, 0.3)
    return mid, r


# ═══════════════════════════════════════════════════════════════
#  动图生成
# ═══════════════════════════════════════════════════════════════

def make_animation(frames, pred_skels, save_path, fps=8, elev=15, azim=-65,
                   action='A27'):
    action_name = ACTION_NAMES.get(action, action)

    fig = plt.figure(figsize=(16, 4.5), facecolor='#0D1117')
    titles = ['GT Skeleton', 'Pred Skeleton (SkelKD)', 'LiDAR Point Cloud', 'mmWave Point Cloud']
    axes   = [fig.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
    for ax, title in zip(axes, titles):
        style_ax(ax, title)
        ax.view_init(elev=elev, azim=azim)

    all_skel  = np.stack([fr['gt_skel'] for fr in frames] + list(pred_skels))
    all_lidar = np.stack([fr['lidar']   for fr in frames])
    all_mm    = np.stack([fr['mmwave']  for fr in frames])

    mid_sk,  r_sk  = _global_range(all_skel)
    mid_li,  r_li  = _global_range(all_lidar)
    mid_mm,  r_mm  = _global_range(all_mm)

    def set_range(ax, mid, r):
        ax.set_xlim(mid[0]-r, mid[0]+r)
        ax.set_ylim(mid[1]-r, mid[1]+r)
        ax.set_zlim(mid[2]-r, mid[2]+r)

    def update(i):
        for ax, title in zip(axes, titles):
            ax.cla()
            style_ax(ax, title)
            ax.view_init(elev=elev, azim=azim)

        fr   = frames[i]
        gt   = fr['gt_skel']
        pred = pred_skels[i].copy(); pred[:, 2] *= -1
        li   = fr['lidar']
        mm   = fr['mmwave']

        draw_skeleton_3d(axes[0], gt)
        set_range(axes[0], mid_sk, r_sk)

        draw_skeleton_3d(axes[1], pred, alpha=0.9)
        set_range(axes[1], mid_sk, r_sk)

        z_li   = li[:, 2]
        z_norm = (z_li - z_li.min()) / (z_li.ptp() + 1e-8)
        axes[2].scatter(li[:,0], li[:,1], li[:,2],
                        c=z_norm, cmap='plasma', s=1.5, alpha=0.7)
        set_range(axes[2], mid_li, r_li)

        axes[3].scatter(mm[:,0], mm[:,1], mm[:,2],
                        c='#FF4444', s=12, alpha=0.9,
                        edgecolors='#FF8888', linewidths=0.3)
        set_range(axes[3], mid_mm, r_mm)

        fig.suptitle(
            f'SKD-Net  |  {action} {action_name}  |  Frame {fr["frame_num"]:03d}  ({i+1}/{len(frames)})',
            color='white', fontsize=11, y=1.01
        )
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000//fps, blit=False
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    ext = os.path.splitext(save_path)[-1].lower()
    if ext == '.gif':
        ani.save(save_path, writer=animation.PillowWriter(fps=fps), dpi=100)
    elif ext == '.mp4':
        ani.save(save_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2000), dpi=120)
    else:
        ani.save(save_path, dpi=100)

    plt.close(fig)
    print(f'  已保存：{save_path}  ({len(frames)}帧, {len(frames)/fps:.1f}s)')


# ═══════════════════════════════════════════════════════════════
#  单动作生成
# ═══════════════════════════════════════════════════════════════

def run_one(args, model, device, subject, action):
    action_name = ACTION_NAMES.get(action, action)
    print(f'\n[{action} {action_name}] subject={subject}')
    try:
        frames = load_sequence(subject, action,
                               start=args.start, end=args.end)
    except RuntimeError as e:
        print(f'  跳过（{e}）')
        return

    pred_skels = predict_skeletons(model, frames, device)

    save_name = f'{action}_{action_name}_{subject}_f{args.start}-{args.end}.{args.fmt}'
    save_path = os.path.join(args.save_dir, save_name)

    make_animation(frames, pred_skels, save_path,
                   fps=args.fps, elev=args.elev, azim=args.azim,
                   action=action)


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def get_parser():
    p = argparse.ArgumentParser('SkelKD 多动作可视化')
    # ── 数据 ──
    p.add_argument('--subject',     default=['S01'], nargs='+', type=str,
                   help='受试者，支持多个：--subject S01 S08')
    p.add_argument('--action',      default='A27',  type=str,
                   help='单个动作，如 A27。使用 --all_actions 可批量生成训练集全部动作')
    p.add_argument('--all_actions', action='store_true',
                   help='批量生成训练集全部 8 个动作（忽略 --action）')
    p.add_argument('--start',       default=0,      type=int)
    p.add_argument('--end',         default=60,     type=int,
                   help='结束帧索引（exclusive），建议 40~80 帧')
    # ── 输出 ──
    p.add_argument('--fps',         default=8,      type=int)
    p.add_argument('--fmt',         default='mp4',  choices=['gif', 'mp4'])
    p.add_argument('--elev',        default=15,     type=int)
    p.add_argument('--azim',        default=-65,    type=int)
    # ── 路径 ──
    p.add_argument('--ckpt',        default=None,   type=str,
                   help='checkpoint 路径，默认自动找最新时间戳目录')
    p.add_argument('--save_dir',    default=None,   type=str,
                   help='保存目录，默认与 --ckpt 同目录')
    p.add_argument('--gpu',         default=0,      type=int)
    return p.parse_args()


def _auto_ckpt():
    dirs = sorted(glob.glob('experiments/SkelKD/mmfi_skelkd_*/'))
    if dirs:
        latest = dirs[-1].rstrip('/')
        return os.path.join(latest, 'checkpt_best.pth'), latest
    p = 'experiments/SkelKD/mmfi_skelkd/checkpt_best.pth'
    return p, 'experiments/SkelKD/mmfi_skelkd'


if __name__ == '__main__':
    args   = get_parser()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.ckpt is None:
        ckpt_path, save_dir = _auto_ckpt()
    else:
        ckpt_path = args.ckpt
        save_dir  = os.path.dirname(ckpt_path)
    if args.save_dir is not None:
        save_dir = args.save_dir
    args.save_dir = save_dir

    print(f'Checkpoint : {ckpt_path}')
    print(f'Save dir   : {save_dir}')

    model = load_model(CFG_PATH, ckpt_path, device)

    # 决定要生成哪些（动作, 受试者）组合
    actions  = TRAIN_ACTIONS if args.all_actions else [args.action]
    subjects = args.subject   # list

    total = len(actions) * len(subjects)
    print(f'\n共 {total} 个视频（{len(actions)} 动作 × {len(subjects)} 受试者）')

    for subject in subjects:
        for action in actions:
            run_one(args, model, device, subject, action)

    print('\n全部完成。')