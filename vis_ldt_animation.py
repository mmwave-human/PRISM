"""
vis_ldt_animation.py
====================
LDT 点云补全结果动图可视化（四格同步）：
  [mmWave Input] | [Generated PC] | [LiDAR GT] | [Skeleton Condition]

用法：
  # 单动作
  python vis_ldt_animation.py \
      --ckpt  experiments/Latent_Diffusion_Trainer/mmfi/checkpt_best.pth \
      --ae_ckpt experiments/Latent_Diffusion_Trainer/mmfi/ae_checkpt_best.pth \
      --action A27 --subject S01 --start 0 --end 60

  # 批量生成训练集全部动作
  python vis_ldt_animation.py \
      --ckpt  experiments/Latent_Diffusion_Trainer/mmfi/checkpt_best.pth \
      --ae_ckpt experiments/Latent_Diffusion_Trainer/mmfi/ae_checkpt_best.pth \
      --all_actions --subject S01 --end 60

  # 批量 + 多受试者
  python vis_ldt_animation.py \
      --ckpt  experiments/Latent_Diffusion_Trainer/mmfi/checkpt_best.pth \
      --ae_ckpt experiments/Latent_Diffusion_Trainer/mmfi/ae_checkpt_best.pth \
      --all_actions --subject S01 S08 --end 60
"""

import argparse
import os
import glob
import re
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.scorenet.score import Score
from model.Compressor.Network import Compressor
from model.AutoEncoderE.Network import AutoEncoderE
from completion_trainer.Latent_SDE_Trainer import Trainer
from tools.io import dict2namespace
import yaml

# ── 常量 ──────────────────────────────────────────────────────
DATA_ROOT = 'data/MMFi/E01'
CFG_PATH  = 'experiments/Latent_Diffusion_Trainer/config.yaml'
MMWAVE_N  = 256
LIDAR_N   = 256

# config.yaml 中训练的动作列表
TRAIN_ACTIONS = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']

# MM-Fi 全部动作名称
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

BONES = [
    (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
BONE_COLORS = {
    (0,1):'#FF6B6B',(0,2):'#FF6B6B',(1,3):'#FF6B6B',(2,4):'#FF6B6B',
    (3,5):'#FF6B6B',(4,6):'#FF6B6B',(5,6):'#FFD93D',
    (5,7):'#6BCB77',(7,9):'#6BCB77',(6,8):'#4D96FF',(8,10):'#4D96FF',
    (5,11):'#FFD93D',(6,12):'#FFD93D',(11,12):'#FFD93D',
    (11,13):'#C77DFF',(13,15):'#C77DFF',(12,14):'#FF9A3C',(14,16):'#FF9A3C',
}

PROTOCOL_ACTIONS = {
    'protocol1': ['A02','A03','A04','A05','A13','A14',
                  'A17','A18','A19','A20','A21','A22','A23','A27'],
    'protocol2': ['A01','A06','A07','A08','A09','A10',
                  'A11','A12','A15','A16','A24','A25','A26'],
    'all':       [f'A{i:02d}' for i in range(1, 28)],
}


def build_action_map(cfg):
    data_cfg = getattr(cfg, 'data', cfg)
    if hasattr(data_cfg, 'actions') and data_cfg.actions:
        actions = list(data_cfg.actions)
    else:
        protocol = getattr(data_cfg, 'protocol', 'protocol1')
        actions  = PROTOCOL_ACTIONS.get(protocol, PROTOCOL_ACTIONS['protocol1'])
    mapping = {act: i for i, act in enumerate(sorted(actions))}
    print(f'  [Label map] {len(mapping)} classes: {mapping}')
    return mapping


# ═══════════════════════════════════════════════════════════════
#  I/O  —— mmWave 使用 TI UART 帧解析（与 MMFiViPC_fixed.py 一致）
# ═══════════════════════════════════════════════════════════════

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0: return np.zeros((1, 3), dtype=np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)

# ── TI IWR6843 UART 帧解析 ─────────────────────────────────────
# Bug 修复：原 _load_mmwave 用 float32 reshape 读的是 UART 帧头（垃圾数据）
# 必须先找 Magic Word，再按 TLV 协议解析实际点云坐标
_MM_MAGIC    = bytes([2, 1, 4, 3, 6, 5, 8, 7])
_MM_HDR_FMT  = "<QIIIIIIII"
_MM_HDR_SIZE = 40

def _try_numpy_float32(path):
    """快速尝试：如果是简单 float32 格式且坐标合理则直接用"""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0: return None
    if raw.size % 5 == 0:
        pc = raw.reshape(-1, 5)
        if np.all(np.abs(pc[:, :3]) < 20): return pc[:, :3]
    if raw.size % 4 == 0:
        pc = raw.reshape(-1, 4)
        if np.all(np.abs(pc[:, :3]) < 20): return pc[:, :3]
    return None

def _parse_ti_uart(path):
    """TI IWR6843 UART 帧格式解析"""
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
                    pts.append([x, y, z]); offset += 16
            elif tlv_type == 7:
                for _ in range(tlv_len // 20):
                    if offset + 20 > len(frame): break
                    x, y, z, v = _struct.unpack_from('<ffff', frame, offset)
                    pts.append([x, y, z]); offset += 20
            else:
                offset += tlv_len
        if pts: all_pts.extend(pts)
        pos = max(idx + _MM_HDR_SIZE, idx + total_len)
    return np.array(all_pts, dtype=np.float32) if all_pts else None

def _load_mmwave(path):
    """
    健壮 mmWave bin 读取，返回 XYZ (N,3)。
    优先 TI UART 解析；快速路径失败时 fallback。
    注意：不做轴变换，与训练代码 MMFiViPC.py 保持完全一致。
    """
    # 优先 TI UART 解析（正确方式）
    pc = _parse_ti_uart(path)
    if pc is not None and len(pc) > 0:
        return pc
    # Fallback：验证过坐标合理性的 float32
    pc = _try_numpy_float32(path)
    if pc is not None and len(pc) > 0:
        return pc
    return np.zeros((1, 3), dtype=np.float32)

def _normalize(pc):
    c = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s = max(np.max(np.sqrt((pc_c ** 2).sum(1))), 1e-8)
    return (pc_c / s).astype(np.float32), c.astype(np.float32), float(s)

def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0: return np.zeros((n, 3), dtype=np.float32)
    idx = np.random.choice(N, n, replace=N < n)
    return pc[idx]


def load_sequence(subject, action, action_map, start=0, end=60):
    lidar_dir  = os.path.join(DATA_ROOT, subject, action, 'lidar')
    mmwave_dir = os.path.join(DATA_ROOT, subject, action, 'mmwave')
    gt_path    = os.path.join(DATA_ROOT, subject, action, 'ground_truth.npy')

    gt_raw = np.load(gt_path, allow_pickle=True)
    if gt_raw.dtype == object: gt_raw = np.array(gt_raw.tolist(), dtype=np.float32)
    else: gt_raw = gt_raw.astype(np.float32)
    s = gt_raw.shape
    if   gt_raw.ndim == 3 and s[1] == 17 and s[2] == 3: gt = gt_raw
    elif gt_raw.ndim == 3 and s[1] == 3  and s[2] == 17: gt = gt_raw.transpose(0, 2, 1)
    elif gt_raw.ndim == 2 and s[1] == 51: gt = gt_raw.reshape(s[0], 17, 3)
    elif gt_raw.ndim == 4 and s[0] == 1:  gt = gt_raw[0].astype(np.float32)
    else: raise ValueError(f'无法解析 GT shape: {gt_raw.shape}')

    l_files = {_frame_num(f): f for f in glob.glob(os.path.join(lidar_dir,  'frame*_filtered.bin'))}
    m_files = {_frame_num(f): f for f in glob.glob(os.path.join(mmwave_dir, 'frame*_filtered.bin'))}
    l_files.pop(-1, None); m_files.pop(-1, None)

    common = sorted(set(l_files) & set(m_files))
    common = [fn for fn in common if fn - 1 < len(gt)][start:end]
    if not common: raise RuntimeError(f'没有找到有效帧：{subject}/{action}')
    print(f'  加载 {subject}/{action}，帧 [{common[0]}, {common[-1]}]，共 {len(common)} 帧')

    label = action_map.get(action, 0)
    frames = []
    for fn in common:
        # ── LiDAR ──────────────────────────────────────────
        lidar = _load_lidar(l_files[fn])
        lidar_norm, shift, scale = _normalize(lidar)
        lidar_vis = _subsample(lidar_norm, LIDAR_N)

        # ── mmWave（TI UART解析，无轴变换，与训练一致）────
        # Bug修复：原版多了一次 [:, [1,0,2]] 轴变换，与训练代码不一致
        mm_xyz = _load_mmwave(m_files[fn])
        valid  = (np.isfinite(mm_xyz).all(1) & (np.abs(mm_xyz) < 20.).all(1))
        mm_xyz = mm_xyz[valid]
        if mm_xyz.shape[0] == 0:
            mm_xyz = np.zeros((1, 3), dtype=np.float32)
        mm_norm, _, _ = _normalize(mm_xyz)
        mm_vis    = _subsample(mm_norm, MMWAVE_N)
        mm_tensor = torch.from_numpy(mm_vis).float().unsqueeze(0)  # (1, N, 3)

        # ── 骨骼：轴变换 + LiDAR shift/scale 归一化（与训练一致）──
        skel_raw    = gt[fn - 1][:, [2, 0, 1]].copy()  # 轴变换对齐 LiDAR
        skel_norm   = (skel_raw - shift) / scale
        skel_tensor = torch.from_numpy(skel_norm.astype(np.float32)).unsqueeze(0)
        skel_vis    = skel_norm.copy(); skel_vis[:, 2] *= -1  # Z取反仅用于显示

        frames.append({
            'lidar_vis':   lidar_vis,
            'mmwave_vis':  mm_vis,
            'mm_tensor':   mm_tensor,
            'gt_skel_vis': skel_vis,
            'skel_tensor': skel_tensor,
            'label':       label,
            'frame_num':   fn,
        })
    return frames


# ═══════════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════════

def load_models(cfg_path, ckpt_path, ae_ckpt_path, device):
    with open(cfg_path) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader))

    model      = Score(cfg.score)
    compressor = Compressor(cfg.compressor)
    trainer    = Trainer(cfg, model=model, compressor=compressor, device=device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint 不存在: {ckpt_path}')
    checkpt = torch.load(ckpt_path, map_location='cpu')
    trainer.model.load_state_dict(checkpt['score_state_dict'], strict=True)
    trainer.compressor.load_state_dict(checkpt['compressor_state_dict'], strict=True)
    trainer.compressor.init()
    trainer.epoch = checkpt.get('epoch', 0) + 1
    trainer.itr   = checkpt.get('itr',   0)

    # EMA swap 修复：防止 sample() 使用随机 EMA shadow 权重
    if hasattr(trainer.optimizer, 'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    trainer.model.eval()
    trainer.compressor.eval()
    print(f'  [Trainer] loaded epoch={checkpt.get("epoch","?")} from {ckpt_path}')

    ae_cfg = getattr(cfg, 'autoencoder_e', None)
    autoencoder_e = AutoEncoderE(
        cfg       = ae_cfg,
        feat_dim  = getattr(ae_cfg, 'feat_dim',   256) if ae_cfg else 256,
        num_coarse= getattr(ae_cfg, 'num_coarse', 128) if ae_cfg else 128,
    ).to(device)
    if ae_ckpt_path and os.path.exists(ae_ckpt_path):
        autoencoder_e.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
        print(f'  [AutoEncoderE] loaded from {ae_ckpt_path}')
    else:
        print(f'  [AutoEncoderE] WARNING: checkpoint 不存在，使用随机初始化')
    autoencoder_e.eval()
    return trainer, autoencoder_e, cfg


# ═══════════════════════════════════════════════════════════════
#  推理
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_sequence(trainer, autoencoder_e, frames, device):
    generated  = []
    total_time = 0.
    for i, fr in enumerate(frames):
        mm    = fr['mm_tensor'].to(device)
        skel  = fr['skel_tensor'].to(device)
        label = torch.tensor([fr['label']], dtype=torch.long).to(device)
        coarse_pts, _ = autoencoder_e(mm)
        t0 = time.time()
        smp_pts, _ = trainer.sample(
            num_samples=1, label=label,
            condition={'skeleton': skel, 'coarse': coarse_pts})
        total_time += time.time() - t0
        generated.append(smp_pts[0].cpu().numpy())
        if (i + 1) % 5 == 0 or i == len(frames) - 1:
            print(f'  推理进度: {i+1}/{len(frames)}  均速: {total_time/(i+1)*1000:.0f} ms/帧')
    return generated


# ═══════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════

def style_ax(ax, title):
    ax.set_facecolor('#0D1117')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#222222')
    ax.tick_params(colors='#444444', labelsize=5)
    ax.set_title(title, color='white', fontsize=10, pad=4)
    ax.set_xlabel('X', color='#444444', fontsize=6, labelpad=1)
    ax.set_ylabel('Y', color='#444444', fontsize=6, labelpad=1)
    ax.set_zlabel('Z', color='#444444', fontsize=6, labelpad=1)

def _global_range(arr_list, pad=1.2):
    pts = np.concatenate([a.reshape(-1, 3) for a in arr_list])
    mn, mx = pts.min(0), pts.max(0)
    mid = (mn + mx) / 2
    r   = max((mx - mn).max() / 2 * pad, 0.3)
    return mid, r


def make_animation(frames, generated, action, save_path,
                   fps=6, elev=15, azim=-65):
    action_name = ACTION_NAMES.get(action, action)

    fig    = plt.figure(figsize=(18, 4.8), facecolor='#0D1117')
    titles = ['mmWave Input', 'Generated PC (LDT)', 'LiDAR GT', 'Skeleton Condition']
    axes   = [fig.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]

    mid_mm,  r_mm  = _global_range([fr['mmwave_vis']  for fr in frames])
    mid_gen, r_gen = _global_range(generated + [fr['lidar_vis'] for fr in frames])
    mid_sk,  r_sk  = _global_range([fr['gt_skel_vis'] for fr in frames])

    zlim_lo = min(a[:, 2].min() for a in generated + [fr['lidar_vis'] for fr in frames])
    zlim_hi = max(a[:, 2].max() for a in generated + [fr['lidar_vis'] for fr in frames])

    def update(i):
        fr, gen = frames[i], generated[i]
        for ax, title in zip(axes, titles):
            ax.cla(); style_ax(ax, title); ax.view_init(elev=elev, azim=azim)

        mm = fr['mmwave_vis']
        axes[0].scatter(mm[:,0], mm[:,1], mm[:,2],
                        c='#FF7700', s=20, alpha=0.95,
                        edgecolors='#FFAA44', linewidths=0.3, depthshade=False)
        axes[0].set_xlim(mid_mm[0]-r_mm, mid_mm[0]+r_mm)
        axes[0].set_ylim(mid_mm[1]-r_mm, mid_mm[1]+r_mm)
        axes[0].set_zlim(mid_mm[2]-r_mm, mid_mm[2]+r_mm)

        axes[1].scatter(gen[:,0], gen[:,1], gen[:,2],
                        c=gen[:,2], cmap='RdYlGn', s=4,
                        vmin=zlim_lo, vmax=zlim_hi, depthshade=False, linewidths=0)
        axes[1].set_xlim(mid_gen[0]-r_gen, mid_gen[0]+r_gen)
        axes[1].set_ylim(mid_gen[1]-r_gen, mid_gen[1]+r_gen)
        axes[1].set_zlim(mid_gen[2]-r_gen, mid_gen[2]+r_gen)

        li = fr['lidar_vis']
        axes[2].scatter(li[:,0], li[:,1], li[:,2],
                        c=li[:,2], cmap='plasma', s=3,
                        vmin=zlim_lo, vmax=zlim_hi, depthshade=False, linewidths=0)
        axes[2].set_xlim(mid_gen[0]-r_gen, mid_gen[0]+r_gen)
        axes[2].set_ylim(mid_gen[1]-r_gen, mid_gen[1]+r_gen)
        axes[2].set_zlim(mid_gen[2]-r_gen, mid_gen[2]+r_gen)

        js = fr['gt_skel_vis']
        for (a, b) in BONES:
            color = BONE_COLORS.get((a, b), BONE_COLORS.get((b, a), '#AAAAAA'))
            axes[3].plot([js[a,0], js[b,0]], [js[a,1], js[b,1]], [js[a,2], js[b,2]],
                         color=color, lw=2)
        axes[3].scatter(js[:,0], js[:,1], js[:,2],
                        c='white', s=12, zorder=5, edgecolors='gray', linewidths=0.5)
        axes[3].set_xlim(mid_sk[0]-r_sk, mid_sk[0]+r_sk)
        axes[3].set_ylim(mid_sk[1]-r_sk, mid_sk[1]+r_sk)
        axes[3].set_zlim(mid_sk[2]-r_sk, mid_sk[2]+r_sk)

        fig.suptitle(
            f'SKD-Net LDT  |  {action} {action_name}  |  Frame {fr["frame_num"]:03d}  ({i+1}/{len(frames)})',
            color='white', fontsize=11, y=1.01)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if save_path.endswith('.mp4'):
        ani.save(save_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2500), dpi=100)
    else:
        ani.save(save_path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f'  已保存：{save_path}  ({len(frames)}帧, {len(frames)/fps:.1f}s)')


# ═══════════════════════════════════════════════════════════════
#  单动作执行
# ═══════════════════════════════════════════════════════════════

def run_one(args, trainer, autoencoder_e, action_map, device, subject, action):
    action_name = ACTION_NAMES.get(action, action)
    print(f'\n[{action} {action_name}] subject={subject}')
    try:
        frames = load_sequence(subject, action, action_map,
                               args.start, args.end)
    except RuntimeError as e:
        print(f'  跳过（{e}）')
        return

    print(f'  开始推理 {len(frames)} 帧...')
    generated = infer_sequence(trainer, autoencoder_e, frames, device)

    save_name = f'ldt_{action}_{action_name}_{subject}_f{args.start}-{args.end}.{args.fmt}'
    save_path = os.path.join(args.save_dir, save_name)
    make_animation(frames, generated, action, save_path,
                   fps=args.fps, elev=args.elev, azim=args.azim)


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LDT 补全结果动图可视化')
    # ── 数据 ──
    parser.add_argument('--data_root',   default=DATA_ROOT)
    parser.add_argument('--subject',     default=['S01'], nargs='+',
                        help='受试者，支持多个：--subject S01 S08')
    parser.add_argument('--action',      default='A27',
                        help='单个动作，如 A27')
    parser.add_argument('--all_actions', action='store_true',
                        help='批量生成训练集全部动作（忽略 --action）')
    parser.add_argument('--start',       type=int, default=0)
    parser.add_argument('--end',         type=int, default=60,
                        help='结束帧索引（exclusive）')
    # ── 模型路径 ──
    parser.add_argument('--cfg',         default=CFG_PATH)
    parser.add_argument('--ckpt',        required=True, help='Trainer checkpoint')
    parser.add_argument('--ae_ckpt',     required=True, help='AutoEncoderE checkpoint')
    # ── 输出 ──
    parser.add_argument('--save_dir',    default='experiments/Latent_Diffusion_Trainer/vis')
    parser.add_argument('--fmt',         default='mp4', choices=['gif', 'mp4'])
    parser.add_argument('--fps',         type=int,   default=6)
    parser.add_argument('--elev',        type=float, default=15)
    parser.add_argument('--azim',        type=float, default=-65)
    parser.add_argument('--gpu',         type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    DATA_ROOT = args.data_root
    trainer, autoencoder_e, cfg = load_models(args.cfg, args.ckpt, args.ae_ckpt, device)
    action_map = build_action_map(cfg)

    # 同步点数常量
    MMWAVE_N = getattr(cfg.data,   'mmwave_npoints', 256)
    LIDAR_N  = getattr(cfg.data,   'tr_max_sample_points', 256)

    actions  = TRAIN_ACTIONS if args.all_actions else [args.action]
    subjects = args.subject

    total = len(actions) * len(subjects)
    print(f'\n共 {total} 个视频（{len(actions)} 动作 × {len(subjects)} 受试者）')

    for subject in subjects:
        for action in actions:
            run_one(args, trainer, autoencoder_e, action_map, device, subject, action)

    print('\n全部完成。')