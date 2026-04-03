#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_mist_infer.py
=================
对 MIST 自采集数据（yaning_shendun2）运行 PRISM 推理并生成动画。

坐标轴一致性说明
─────────────────────────────────────────────────────────────────
传感器原生坐标（GUI 采集存储格式）：
  LiDAR  .bin : (lateral, depth, height)   — MID-360 原生
  mmWave .npy : (X_lat, Y_dep, Z_hgt, Doppler, SNR) — IWR6843 原生

MMFi 训练数据约定（模型期望输入）：
  点云 : (depth, lateral, height)   col0=depth ≈ 1~3 m

GT 标注约定（h36m_annotator.py 输出）：
  ground_truth_h36m.npy : (lateral, height_up, depth)
    col0=lateral, col1=height(up+), col2=depth

变换流程：
  LiDAR  : raw[:,[1,0,2]] → (dep, lat, hgt) = MMFi ✓
  mmWave : xyz→raw_xyz[:,[1,0,2]] → (dep, lat, hgt) = MMFi ✓
  GT     : gt[fn][:,[2,0,1]] → (dep, lat, hgt) = pointcloud 空间 ✓

动作标签（squat A12 在 8-action 训练集中排序索引）：
  sorted([A03,A12,A13,A17,A19,A22,A26,A27]) → A12 = index 1
─────────────────────────────────────────────────────────────────

用法：
  python vis_mist_infer.py [--start 0] [--end 80] [--fps 8] [--fmt mp4]
      [--cfg  experiments/Latent_Diffusion_Trainer/config_8actions.yaml]
      [--ckpt experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_3600.pth]
      [--ae   experiments/Latent_Diffusion_Trainer/mmfi_202603151358/autoencoder_e_3600.pth]
      [--data assets/DataCollection/collected_data/yaning_shendun2]
      [--out  outputs/mist_infer]
"""

import os, sys, glob, re, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.gridspec import GridSpec
import yaml
try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from completion_trainer.Latent_SDE_Trainer import Trainer
from model.scorenet.score import Score
from model.Compressor.Network import Compressor
from model.GeoAE.Network import GeoAE
from tools.io import dict2namespace

# ─────────────────────────────────────────────────────────────────
#  H36M-17 骨骼定义（与 vis_ldt_animation.py 一致）
# ─────────────────────────────────────────────────────────────────
BONES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]
BONE_COLORS = {
    (0,1):'#FF6B6B',(1,2):'#FF6B6B',(2,3):'#FF6B6B',
    (0,4):'#48BB78',(4,5):'#48BB78',(5,6):'#48BB78',
    (0,7):'#DDDDDD',(7,8):'#DDDDDD',(8,9):'#DDDDDD',(9,10):'#DDDDDD',
    (8,11):'#4D96FF',(11,12):'#4D96FF',(12,13):'#4D96FF',
    (8,14):'#FFD700',(14,15):'#FFD700',(15,16):'#FFD700',
}

# action label: A12 = index 1 in sorted 8-action list
SQUAT_LABEL = 1

# RealSense D435i 默认内参（640×480）
_FX, _FY, _CX, _CY = 617.0, 617.0, 320.0, 240.0

# ─────────────────────────────────────────────────────────────────
#  I/O
# ─────────────────────────────────────────────────────────────────

def _frame_num(p):
    m = re.search(r'frame(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else -1


def load_lidar_raw(path):
    """Load raw MID-360 .bin: stored as float64, shape (N,3) = (lat, dep, hgt)."""
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)


def load_mmwave_raw(path):
    """Load raw IWR6843 .npy: float32 (N,5) = (X_lat, Y_dep, Z_hgt, Doppler, SNR)."""
    try:
        pc = np.load(path).astype(np.float32)
        if pc.ndim == 2 and pc.shape[1] >= 3:
            if pc.shape[1] < 5:
                pc = np.hstack([pc, np.zeros((len(pc), 5 - pc.shape[1]), np.float32)])
            return pc[:, :5]
    except Exception:
        pass
    return np.zeros((0, 5), dtype=np.float32)


def to_mmfi_lidar(raw, gt_centroid_dep_lat_hgt, margin=1.2):
    """
    raw: (N,3) = (lat, dep, hgt)  →  output (M,3) = (dep, lat, hgt)

    人体中心裁剪：以GT骨架质心为中心，保留 margin 米半径内的点。
    这样避免把整个房间的点云送入归一化，保证 shift/scale 反映人体尺寸。
    """
    if len(raw) == 0:
        return raw
    # Swap axes: (lat, dep, hgt) → (dep, lat, hgt)
    pc = raw[:, [1, 0, 2]]
    # Crop to person bounding box (in MMFi space)
    cx, cy, cz = gt_centroid_dep_lat_hgt
    mask = ((np.abs(pc[:, 0] - cx) < margin) &
            (np.abs(pc[:, 1] - cy) < margin * 0.8) &
            (np.abs(pc[:, 2] - cz) < margin))
    result = pc[mask]
    if len(result) < 10:
        # Fallback: looser crop if too few points
        mask2 = ((np.abs(pc[:, 0] - cx) < margin * 2) &
                 (np.abs(pc[:, 1] - cy) < margin * 1.5) &
                 (np.abs(pc[:, 2] - cz) < margin * 2))
        result = pc[mask2]
    return result if len(result) > 0 else pc


def to_mmfi_mmwave(raw5):
    """
    raw5: (N,5) = (X_lat, Y_dep, Z_hgt, Doppler, SNR)  →  output (M,3) = (dep, lat, hgt)

    过滤无效点，仅保留合理范围（人体尺度）。
    """
    if len(raw5) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # Basic validity filter on original axes
    x_lat = raw5[:, 0]
    y_dep = raw5[:, 1]
    z_hgt = raw5[:, 2]
    valid = (np.isfinite(x_lat) & np.isfinite(y_dep) & np.isfinite(z_hgt) &
             (y_dep > 0.1) & (y_dep < 8.0) &
             (np.abs(x_lat) < 4.0) &
             (np.abs(z_hgt) < 3.0))
    filtered = raw5[valid]
    if len(filtered) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # Swap: (X_lat, Y_dep, Z_hgt) → (Y_dep, X_lat, Z_hgt) = (dep, lat, hgt)
    return filtered[:, [1, 0, 2]]


# ─────────────────────────────────────────────────────────────────
#  RGB 与 2D 投影
# ─────────────────────────────────────────────────────────────────

def load_rgb_frame(path):
    """返回 (H, W, 3) uint8 RGB 数组；文件不存在时返回 None。"""
    if not os.path.isfile(path):
        return None
    if _HAS_PIL:
        return np.array(_PIL_Image.open(path).convert('RGB'))
    # fallback: matplotlib（PNG 返回 float32 0-1，需转为 uint8）
    arr = plt.imread(path)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr[..., :3]


def project_joints_2d(gt_lat_hgt_dep, fx, fy, cx, cy):
    """
    GT 格式 (lat, hgt, dep) → 像素坐标 (u, v)

    相机坐标换算：
      cam_X = lateral = gt[:,0]
      cam_Y = -height_up = -gt[:,1]   (相机 Y 向下，height 向上)
      cam_Z = depth = gt[:,2]
    投影：
      u = fx * cam_X / cam_Z + cx
      v = fy * cam_Y / cam_Z + cy
    """
    lat = gt_lat_hgt_dep[:, 0]
    hgt = gt_lat_hgt_dep[:, 1]
    dep = gt_lat_hgt_dep[:, 2]
    valid = dep > 0.05
    u = np.full(17, np.nan, np.float32)
    v = np.full(17, np.nan, np.float32)
    u[valid] = fx * lat[valid] / dep[valid] + cx
    v[valid] = fy * (-hgt[valid]) / dep[valid] + cy
    return u, v   # pixel coords; v increases downward


def draw_skeleton_on_rgb(ax, rgb_img, u, v, bones, bone_colors, img_wh=(640, 480)):
    """在 ax 上用 imshow 显示 RGB，再叠加 H36M-17 2D 骨骼。"""
    W, H = img_wh
    ax.imshow(rgb_img, aspect='auto', extent=[0, W, H, 0])  # y 轴：0 在顶
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis('off')

    # 骨骼连线
    for (a, b) in bones:
        if np.isfinite(u[a]) and np.isfinite(u[b]):
            col = bone_colors.get((a, b), bone_colors.get((b, a), '#AAAAAA'))
            ax.plot([u[a], u[b]], [v[a], v[b]], color=col, lw=2.0, alpha=0.85)

    # 关节点
    valid = np.isfinite(u) & np.isfinite(v)
    if valid.any():
        ax.scatter(u[valid], v[valid], c='white', s=18, zorder=5,
                   edgecolors='gray', linewidths=0.5)


# ─────────────────────────────────────────────────────────────────
#  归一化
# ─────────────────────────────────────────────────────────────────

def _normalize(pc):
    c = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s = max(float(np.max(np.sqrt((pc_c ** 2).sum(1)))), 1e-8)
    return (pc_c / s).astype(np.float32), c.astype(np.float32), s


def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = np.random.choice(N, n, replace=(N < n))
    return pc[idx]


# ─────────────────────────────────────────────────────────────────
#  数据加载
# ─────────────────────────────────────────────────────────────────

def load_sequence(data_dir, start=0, end=80, mmwave_n=256, lidar_n=256,
                  fx=_FX, fy=_FY, cx=_CX, cy=_CY):
    """
    加载 yaning_shendun2 原始数据，返回逐帧字典列表。

    坐标一致性验证：
      - gt_raw 储存格式  (lat, hgt, dep)  →  [:,[2,0,1]] → (dep, lat, hgt)
      - LiDAR 经 to_mmfi_lidar   → (dep, lat, hgt)
      - mmWave 经 to_mmfi_mmwave → (dep, lat, hgt)
      以上三者在同一坐标系，可直接叠加可视化。
    """
    gt_path    = os.path.join(data_dir, 'ground_truth_h36m.npy')
    lidar_dir  = os.path.join(data_dir, 'lidar')
    mmwave_dir = os.path.join(data_dir, 'mmwave')
    rgb_dir    = os.path.join(data_dir, 'rgb')

    # ── Ground Truth ────────────────────────────────────────────
    gt_raw = np.load(gt_path).astype(np.float32)   # (T, 17, 3) = (lat, hgt, dep)
    T = gt_raw.shape[0]
    print(f'  GT loaded: {T} frames, shape={gt_raw.shape}')
    print(f'  GT convention: col0=lateral {gt_raw[:,:,0].mean():.3f}m  '
          f'col1=height {gt_raw[:,:,1].mean():.3f}m  '
          f'col2=depth {gt_raw[:,:,2].mean():.3f}m')

    # ── 帧索引 ──────────────────────────────────────────────────
    l_files = sorted(glob.glob(os.path.join(lidar_dir,  'frame*.bin')))
    m_files = sorted(glob.glob(os.path.join(mmwave_dir, 'frame*.npy')))
    r_files = sorted(glob.glob(os.path.join(rgb_dir,    'frame*.png')))
    l_map = {_frame_num(f): f for f in l_files}
    m_map = {_frame_num(f): f for f in m_files}
    r_map = {_frame_num(f): f for f in r_files}
    common_raw = sorted(set(l_map) & set(m_map))
    # frame_num → gt index (0-based)
    common = [fn for fn in common_raw if fn < T][start:end]
    if not common:
        raise RuntimeError(f'没有有效帧（帧索引超出GT长度 {T}）')
    print(f'  使用帧范围: [{common[0]}, {common[-1]}]，共 {len(common)} 帧')

    frames = []
    for fn in common:
        # ── GT 骨骼 ────────────────────────────────────────────
        # gt_raw[fn]: (17,3) = (lat, hgt, dep)
        # 变换到点云空间: (lat,hgt,dep) → (dep,lat,hgt) via [:,[2,0,1]]
        skel_gt = gt_raw[fn][:, [2, 0, 1]].copy()   # (17,3) = (dep, lat, hgt)

        # GT 质心（在 MMFi/pointcloud 空间）
        gt_centroid = skel_gt.mean(axis=0)           # (dep, lat, hgt)

        # ── LiDAR ─────────────────────────────────────────────
        lidar_raw = load_lidar_raw(l_map[fn])
        lidar_mmfi = to_mmfi_lidar(lidar_raw, gt_centroid)   # (dep, lat, hgt)
        if len(lidar_mmfi) < 5:
            lidar_mmfi = np.tile(skel_gt, (10, 1))  # degenerate fallback

        lidar_norm, shift, scale = _normalize(lidar_mmfi)
        lidar_vis = _subsample(lidar_norm, lidar_n)

        # ── mmWave ─────────────────────────────────────────────
        mm_raw5 = load_mmwave_raw(m_map[fn])
        mm_xyz = to_mmfi_mmwave(mm_raw5)              # (dep, lat, hgt)
        if len(mm_xyz) == 0:
            mm_xyz = np.zeros((1, 3), dtype=np.float32)

        mm_norm, _, _ = _normalize(mm_xyz)
        mm_vis    = _subsample(mm_norm, mmwave_n)
        mm_tensor = torch.from_numpy(mm_vis).float().unsqueeze(0)

        # ── 骨骼归一化（与 LiDAR shift/scale 对齐）─────────────
        skel_norm = (skel_gt - shift) / scale
        skel_tensor = torch.from_numpy(skel_norm.astype(np.float32)).unsqueeze(0)
        skel_vis = skel_norm.copy()
        skel_vis[:, 2] *= -1    # matplotlib Z 向上翻转（与 vis_ldt_animation 一致）

        # ── RGB + 2D 骨骼投影 ──────────────────────────────────
        # gt_raw[fn] 格式 (lat, hgt, dep) → 直接投影，无需坐标系变换
        skel_2d_u, skel_2d_v = project_joints_2d(gt_raw[fn], fx, fy, cx, cy)
        rgb_path = r_map.get(fn, None)

        frames.append({
            'lidar_vis':   lidar_vis,
            'mmwave_vis':  mm_vis,
            'mm_tensor':   mm_tensor,
            'gt_skel_vis': skel_vis,
            'skel_tensor': skel_tensor,
            'frame_num':   fn,
            'n_mm_pts':    len(mm_xyz),
            'n_lidar_pts': len(lidar_mmfi),
            'rgb_path':    rgb_path,
            'skel_2d_u':   skel_2d_u,
            'skel_2d_v':   skel_2d_v,
        })

    # ── 数据质量汇报 ────────────────────────────────────────────
    mm_counts = [f['n_mm_pts'] for f in frames]
    li_counts = [f['n_lidar_pts'] for f in frames]
    print(f'  mmWave pts/frame:  mean={np.mean(mm_counts):.1f}  '
          f'min={min(mm_counts)}  max={max(mm_counts)}')
    print(f'  LiDAR pts/frame:   mean={np.mean(li_counts):.1f}  '
          f'min={min(li_counts)}  max={max(li_counts)}')
    return frames


# ─────────────────────────────────────────────────────────────────
#  模型加载（与 vis_ldt_animation.py 完全相同）
# ─────────────────────────────────────────────────────────────────

def load_models(cfg_path, ckpt_path, ae_ckpt_path, device):
    with open(cfg_path) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader))

    model      = Score(cfg.score)
    compressor = Compressor(cfg.compressor)
    trainer    = Trainer(cfg, model=model, compressor=compressor, device=device)

    checkpt = torch.load(ckpt_path, map_location='cpu')
    trainer.model.load_state_dict(checkpt['score_state_dict'], strict=True)
    trainer.compressor.load_state_dict(checkpt['compressor_state_dict'], strict=True)
    trainer.compressor.init()
    trainer.epoch = checkpt.get('epoch', 0) + 1

    if hasattr(trainer.optimizer, 'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    trainer.model.eval()
    trainer.compressor.eval()
    print(f'  [Score/Compressor] epoch={checkpt.get("epoch","?")}  {ckpt_path}')

    ae_cfg = getattr(cfg, 'geo_ae', None)
    geo_ae = GeoAE(
        cfg        = ae_cfg,
        feat_dim   = getattr(ae_cfg, 'feat_dim',   256) if ae_cfg else 256,
        num_coarse = getattr(ae_cfg, 'num_coarse', 128) if ae_cfg else 128,
    ).to(device)

    if ae_ckpt_path and os.path.exists(ae_ckpt_path):
        geo_ae.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
        print(f'  [GeoAE] {ae_ckpt_path}')
    else:
        print(f'  [GeoAE] WARNING: checkpoint 未找到，使用随机初始化: {ae_ckpt_path}')
    geo_ae.eval()

    return trainer, geo_ae, cfg


# ─────────────────────────────────────────────────────────────────
#  推理（与 vis_ldt_animation.py 相同）
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_sequence(trainer, geo_ae, frames, device):
    generated  = []
    total_time = 0.

    label_tensor = torch.tensor([SQUAT_LABEL], dtype=torch.long).to(device)
    print(f'  动作标签: A12 Squat = {SQUAT_LABEL}')

    for i, fr in enumerate(frames):
        mm   = fr['mm_tensor'].to(device)
        skel = fr['skel_tensor'].to(device)

        coarse_pts, _ = geo_ae(mm)

        t0 = time.time()
        smp_pts, _ = trainer.sample(
            num_samples=1,
            label=label_tensor,
            condition={'skeleton': skel, 'coarse': coarse_pts})
        total_time += time.time() - t0

        generated.append(smp_pts[0].cpu().numpy())

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f'  推理进度: {i+1}/{len(frames)}'
                  f'  均速: {total_time/(i+1)*1000:.0f} ms/帧')

    return generated


# ─────────────────────────────────────────────────────────────────
#  绘图
# ─────────────────────────────────────────────────────────────────

def style_ax(ax, title):
    ax.set_facecolor('#0D1117')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False
        p.set_edgecolor('#222222')
    ax.tick_params(colors='#444444', labelsize=5)
    ax.set_title(title, color='white', fontsize=10, pad=4)
    ax.set_xlabel('depth', color='#555', fontsize=6, labelpad=1)
    ax.set_ylabel('lateral', color='#555', fontsize=6, labelpad=1)
    ax.set_zlabel('height', color='#555', fontsize=6, labelpad=1)


def _global_range(arr_list, pad=1.2):
    pts = np.concatenate([a.reshape(-1, 3) for a in arr_list])
    mn, mx = pts.min(0), pts.max(0)
    mid = (mn + mx) / 2
    r   = max((mx - mn).max() / 2 * pad, 0.3)
    return mid, r


def make_animation(frames, generated, save_path, fps=8, elev=15, azim=-65):
    """
    5 列布局（左→右）：
      0: RGB + 2D H36M 骨骼叠加
      1: mmWave 输入 (3D)
      2: PRISM 生成点云 (3D, viridis 高度着色)
      3: LiDAR GT (3D, plasma 高度着色)
      4: 3D 骨骼条件 (3D)
    """
    # ── 布局：RGB 列比 3D 列略宽 ──────────────────────────────
    fig = plt.figure(figsize=(25, 5), facecolor='#0D1117')
    gs  = GridSpec(1, 5, figure=fig,
                   width_ratios=[1.35, 1, 1, 1, 1],
                   wspace=0.08, left=0.02, right=0.98,
                   top=0.88, bottom=0.05)

    ax_rgb = fig.add_subplot(gs[0])               # 2D image
    ax_mm  = fig.add_subplot(gs[1], projection='3d')
    ax_gen = fig.add_subplot(gs[2], projection='3d')
    ax_li  = fig.add_subplot(gs[3], projection='3d')
    ax_sk  = fig.add_subplot(gs[4], projection='3d')

    for ax, title in [(ax_mm,  'mmWave Input'),
                      (ax_gen, 'PRISM Output'),
                      (ax_li,  'LiDAR GT'),
                      (ax_sk,  'Skeleton Condition')]:
        ax.set_title(title, color='white', fontsize=9, pad=3)

    ax_rgb.set_title('RGB + H36M-17 Skeleton', color='white', fontsize=9, pad=3)

    # ── 预计算全局坐标范围（固定轴）────────────────────────────
    mid_mm,  r_mm  = _global_range([fr['mmwave_vis'] for fr in frames])
    mid_gen, r_gen = _global_range(generated + [fr['lidar_vis'] for fr in frames])
    mid_sk,  r_sk  = _global_range([fr['gt_skel_vis'] for fr in frames])

    zlim_lo = min(a[:, 2].min() for a in generated + [fr['lidar_vis'] for fr in frames])
    zlim_hi = max(a[:, 2].max() for a in generated + [fr['lidar_vis'] for fr in frames])

    # ── 预加载所有 RGB 帧（避免每帧磁盘 IO 阻塞动画）──────────
    _blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_cache  = {}
    for fr in frames:
        fn = fr['frame_num']
        if fr['rgb_path'] and fn not in rgb_cache:
            img = load_rgb_frame(fr['rgb_path'])
            rgb_cache[fn] = img if img is not None else _blank_rgb

    def _style3d(ax):
        ax.set_facecolor('#0D1117')
        for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            p.fill = False
            p.set_edgecolor('#1e1e1e')
        ax.tick_params(colors='#333', labelsize=4)

    def update(i):
        fr, gen = frames[i], generated[i]

        # ── Panel 0: RGB + 2D skeleton ─────────────────────────
        ax_rgb.cla()
        ax_rgb.set_facecolor('#0D1117')
        ax_rgb.set_title('RGB + H36M-17 Skeleton', color='white', fontsize=9, pad=3)
        rgb_img = rgb_cache.get(fr['frame_num'], _blank_rgb)
        H, W = rgb_img.shape[:2]
        draw_skeleton_on_rgb(ax_rgb, rgb_img,
                             fr['skel_2d_u'], fr['skel_2d_v'],
                             BONES, BONE_COLORS, img_wh=(W, H))

        # ── Panel 1: mmWave input ───────────────────────────────
        ax_mm.cla(); _style3d(ax_mm)
        ax_mm.set_title('mmWave Input', color='white', fontsize=9, pad=3)
        ax_mm.view_init(elev=elev, azim=azim)
        mm = fr['mmwave_vis']
        ax_mm.scatter(mm[:, 0], mm[:, 1], mm[:, 2],
                      c='#FF7700', s=22, alpha=0.95,
                      edgecolors='#FFAA44', linewidths=0.3, depthshade=False)
        ax_mm.set_xlim(mid_mm[0]-r_mm, mid_mm[0]+r_mm)
        ax_mm.set_ylim(mid_mm[1]-r_mm, mid_mm[1]+r_mm)
        ax_mm.set_zlim(mid_mm[2]-r_mm, mid_mm[2]+r_mm)
        ax_mm.text2D(0.5, -0.03, f'{fr["n_mm_pts"]} pts',
                     transform=ax_mm.transAxes, color='#666', fontsize=7, ha='center')

        # ── Panel 2: PRISM output ───────────────────────────────
        ax_gen.cla(); _style3d(ax_gen)
        ax_gen.set_title('PRISM Output', color='white', fontsize=9, pad=3)
        ax_gen.view_init(elev=elev, azim=azim)
        ax_gen.scatter(gen[:, 0], gen[:, 1], gen[:, 2],
                       c=gen[:, 2], cmap='viridis', s=4,
                       vmin=zlim_lo, vmax=zlim_hi,
                       depthshade=False, linewidths=0)
        ax_gen.set_xlim(mid_gen[0]-r_gen, mid_gen[0]+r_gen)
        ax_gen.set_ylim(mid_gen[1]-r_gen, mid_gen[1]+r_gen)
        ax_gen.set_zlim(mid_gen[2]-r_gen, mid_gen[2]+r_gen)

        # ── Panel 3: LiDAR GT ───────────────────────────────────
        ax_li.cla(); _style3d(ax_li)
        ax_li.set_title('LiDAR GT', color='white', fontsize=9, pad=3)
        ax_li.view_init(elev=elev, azim=azim)
        li = fr['lidar_vis']
        ax_li.scatter(li[:, 0], li[:, 1], li[:, 2],
                      c=li[:, 2], cmap='plasma', s=3,
                      vmin=zlim_lo, vmax=zlim_hi,
                      depthshade=False, linewidths=0)
        ax_li.set_xlim(mid_gen[0]-r_gen, mid_gen[0]+r_gen)
        ax_li.set_ylim(mid_gen[1]-r_gen, mid_gen[1]+r_gen)
        ax_li.set_zlim(mid_gen[2]-r_gen, mid_gen[2]+r_gen)

        # ── Panel 4: 3D Skeleton ────────────────────────────────
        ax_sk.cla(); _style3d(ax_sk)
        ax_sk.set_title('Skeleton Condition', color='white', fontsize=9, pad=3)
        ax_sk.view_init(elev=elev, azim=azim)
        js = fr['gt_skel_vis']
        for (a, b) in BONES:
            col = BONE_COLORS.get((a, b), BONE_COLORS.get((b, a), '#AAAAAA'))
            ax_sk.plot([js[a, 0], js[b, 0]],
                       [js[a, 1], js[b, 1]],
                       [js[a, 2], js[b, 2]], color=col, lw=2)
        ax_sk.scatter(js[:, 0], js[:, 1], js[:, 2],
                      c='white', s=15, zorder=5, edgecolors='gray', linewidths=0.5)
        ax_sk.set_xlim(mid_sk[0]-r_sk, mid_sk[0]+r_sk)
        ax_sk.set_ylim(mid_sk[1]-r_sk, mid_sk[1]+r_sk)
        ax_sk.set_zlim(mid_sk[2]-r_sk, mid_sk[2]+r_sk)

        fig.suptitle(
            f'PRISM  ·  MIST yaning_shendun2 (Squat / A12)'
            f'  ·  Frame {fr["frame_num"]:03d}  ({i+1}/{len(frames)})',
            color='#cccccc', fontsize=10, y=0.98)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    ani.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f'  已保存: {save_path}  ({len(frames)}帧, {len(frames)/fps:.1f}s)')


# ─────────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser('MIST 自采集数据 PRISM 推理可视化')
    parser.add_argument('--data', default='assets/DataCollection/collected_data/yaning_shendun2')
    parser.add_argument('--cfg',  default='experiments/Latent_Diffusion_Trainer/config_8actions.yaml')
    parser.add_argument('--ckpt', default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_3600.pth')
    parser.add_argument('--ae',   default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/autoencoder_e_3600.pth')
    parser.add_argument('--out',  default='outputs/mist_infer')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end',   type=int, default=80)
    parser.add_argument('--fps',   type=int, default=8)
    parser.add_argument('--fmt',   default='mp4')
    parser.add_argument('--elev',  type=float, default=15)
    parser.add_argument('--azim',  type=float, default=-65)
    parser.add_argument('--fx',    type=float, default=_FX, help='相机内参 fx')
    parser.add_argument('--fy',    type=float, default=_FY, help='相机内参 fy')
    parser.add_argument('--cx',    type=float, default=_CX, help='相机内参 cx')
    parser.add_argument('--cy',    type=float, default=_CY, help='相机内参 cy')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'Device: {device}')

    # ── 验证坐标轴一致性（仅统计，不影响推理）──────────────────
    print('\n=== 坐标轴一致性验证 ===')
    gt = np.load(os.path.join(args.data, 'ground_truth_h36m.npy'))
    print(f'GT (lat,hgt,dep) col0_lat={gt[:,:,0].mean():.3f}m  '
          f'col1_hgt={gt[:,:,1].mean():.3f}m  col2_dep={gt[:,:,2].mean():.3f}m')
    gt_mmfi = gt[:, :, [2, 0, 1]]  # → (dep, lat, hgt)
    print(f'GT after [:,[2,0,1]] → (dep,lat,hgt)  '
          f'col0_dep_mean={gt_mmfi[:,:,0].mean():.3f}m  ← 应 > 1m')

    # Load one lidar frame as sanity check
    l_files = sorted(glob.glob(os.path.join(args.data, 'lidar', 'frame*.bin')))
    sample_lidar = np.fromfile(l_files[50], dtype=np.float64).reshape(-1,3).astype(np.float32)
    # person centroid from GT
    gt_centroid = gt_mmfi[50].mean(axis=0)
    lidar_check = to_mmfi_lidar(sample_lidar, gt_centroid)
    print(f'LiDAR sample (dep,lat,hgt):  '
          f'dep_mean={lidar_check[:,0].mean():.3f}m  '
          f'lat_mean={lidar_check[:,1].mean():.3f}m  '
          f'hgt_mean={lidar_check[:,2].mean():.3f}m')
    print(f'  GT depth ≈ {gt_centroid[0]:.3f}m  LiDAR depth ≈ {lidar_check[:,0].mean():.3f}m  '
          f'← 两者应接近 ✓' if abs(gt_centroid[0] - lidar_check[:,0].mean()) < 0.5 else
          f'  !! 不匹配，请检查坐标轴 !!')
    print('========================\n')

    # ── 加载数据 ────────────────────────────────────────────────
    print('[1/3] 加载序列数据...')
    frames = load_sequence(args.data, args.start, args.end,
                           fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    # ── 加载模型 ────────────────────────────────────────────────
    print('\n[2/3] 加载模型...')
    trainer, geo_ae, cfg = load_models(args.cfg, args.ckpt, args.ae, device)

    # ── 推理 ────────────────────────────────────────────────────
    print('\n[3/3] 推理中...')
    generated = infer_sequence(trainer, geo_ae, frames, device)

    # ── 保存 ────────────────────────────────────────────────────
    fname = f'mist_shendun2_f{args.start}-{args.end}.{args.fmt}'
    save_path = os.path.join(args.out, fname)
    print(f'\n[保存] {save_path}')
    make_animation(frames, generated, save_path,
                   fps=args.fps, elev=args.elev, azim=args.azim)
    print('\n完成。')


if __name__ == '__main__':
    main()
