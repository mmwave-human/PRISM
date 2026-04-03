#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_demo.py — PRISM Demo 视频生成器（2×3 布局，白底，物理坐标）

布局
────────────────────────────────────────────────────────────────────────
  行0: [RGB + 2D骨骼]  [深度图]        [LiDAR GT 点云]
  行1: [mmWave 稀疏]   [3D 骨骼]       [PRISM 生成点云]
       ← MIST →        ←────── MMFi E01/S07/A12 (Squat) ──────────────→

坐标约定（物理空间，全程一致）
────────────────────────────────────────────────────────────────────────
  LiDAR / mmWave / PRISM / Skeleton：均为 (dep, lat, hgt)
    col0 = depth  ≈ 2.9~3.2 m
    col1 = lateral（左右）
    col2 = height（向上为正，在传感器坐标系中）
  GT skeleton 原始: (lat, hgt_up, dep) → [:,[2,0,1]] → (dep, lat, hgt_up)
  mmWave 文件格式: float64 N×5 = (dep, lat, hgt, vel, snr)，与 LiDAR 一致
  PRISM 生成: 归一化空间 → 反归一化到物理坐标显示

用法
────────────────────────────────────────────────────────────────────────
  python vis_demo.py [--start 0] [--end 80] [--fps 8] [--device cuda]
"""

import os, sys, glob, re, time, argparse, struct
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa
import yaml

try:
    from PIL import Image as _PIL
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
#  常量
# ─────────────────────────────────────────────────────────────────

ACTION_LABEL = 7   # A27 (Bowing) 在 sorted 8-action list 中的下标
FX, FY, CX, CY = 617.0, 617.0, 320.0, 240.0

BONES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]
_BC = {
    (0,1):'#CC3333',(1,2):'#CC3333',(2,3):'#CC3333',
    (0,4):'#2E8B57',(4,5):'#2E8B57',(5,6):'#2E8B57',
    (0,7):'#555555',(7,8):'#555555',(8,9):'#555555',(9,10):'#555555',
    (8,11):'#1E6FCC',(11,12):'#1E6FCC',(12,13):'#1E6FCC',
    (8,14):'#B8860B',(14,15):'#B8860B',(15,16):'#B8860B',
}

# TI UART 解析（CoordVerifier 一致，作为 mmwave 读取的备用路径）
_MAGIC    = bytes([2, 1, 4, 3, 6, 5, 8, 7])
_HDR_FMT  = "<QIIIIIIII"
_HDR_SIZE = 40

# ─────────────────────────────────────────────────────────────────
#  mmWave I/O（优先 float64，与 LiDAR 格式一致）
# ─────────────────────────────────────────────────────────────────

def _try_numpy_float32_whole(path):
    """整体数组检查（CoordVerifier 逻辑）。"""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return None
    if raw.size % 5 == 0:
        pc = raw.reshape(-1, 5)
        if np.all(np.abs(pc[:, :3]) < 20):
            return pc[:, :3]
    if raw.size % 4 == 0:
        pc = raw.reshape(-1, 4)
        if np.all(np.abs(pc[:, :3]) < 20):
            return pc[:, :3]
    return None


def _parse_ti_uart(path):
    """TI UART 帧解析（CoordVerifier 一致）。"""
    raw = open(path, "rb").read()
    all_pts, pos = [], 0
    while True:
        idx = raw.find(_MAGIC, pos)
        if idx == -1:
            break
        if idx + _HDR_SIZE > len(raw):
            break
        total_len = struct.unpack_from("<I", raw, idx + 12)[0]
        frame_raw = raw[idx: idx + total_len]
        if len(frame_raw) < _HDR_SIZE:
            pos = idx + _HDR_SIZE
            continue
        try:
            header = struct.unpack_from(_HDR_FMT, frame_raw, 0)
        except struct.error:
            pos = idx + _HDR_SIZE
            continue
        if header[0] != struct.unpack("<Q", _MAGIC)[0]:
            pos = idx + _HDR_SIZE
            continue
        pts, offset = [], _HDR_SIZE
        for _ in range(header[7]):
            if offset + 8 > len(frame_raw):
                break
            tlv_type, tlv_len = struct.unpack_from("<II", frame_raw, offset)
            offset += 8
            if tlv_type == 1:
                for _ in range(tlv_len // 16):
                    if offset + 16 > len(frame_raw):
                        break
                    x, y, z, vel = struct.unpack_from("<ffff", frame_raw, offset)
                    pts.append([x, y, z])
                    offset += 16
            elif tlv_type == 7:
                for _ in range(tlv_len // 20):
                    if offset + 20 > len(frame_raw):
                        break
                    x, y, z, vel = struct.unpack_from("<ffff", frame_raw, offset)
                    pts.append([x, y, z])
                    offset += 20
            else:
                offset += tlv_len
        if pts:
            all_pts.append(np.array(pts, dtype=np.float32))
        pos = max(idx + _HDR_SIZE, idx + total_len)
    return np.vstack(all_pts) if all_pts else None


def _mm_filter(pc):
    """
    保留物理上合理的点：
      dep (col0) > 0.5 m
      |lat (col1)| < 3 m
      hgt (col2) in (-2.0, 2.5) m
    """
    m = (np.isfinite(pc).all(1) &
         (np.abs(pc) < 20).all(1) &
         (pc[:, 0] > 0.5) &
         (np.abs(pc[:, 1]) < 3.0) &
         (pc[:, 2] > -2.0) & (pc[:, 2] < 2.5))
    return pc[m]


def load_mmwave_mmfi(path):
    """
    MMFi mmwave: float64 N×5 = (dep, lat, hgt, vel, snr)，与 LiDAR 轴序一致。
    优先级：float32 整体检查 → TI UART → float64（主要路径）
    返回 (N,3) float32 = (dep, lat, hgt) 物理坐标。
    """
    # 1. float32 整体检查
    pc = _try_numpy_float32_whole(path)
    if pc is not None and len(pc) > 0:
        return _mm_filter(pc.astype(np.float32))

    # 2. TI UART
    pc = _parse_ti_uart(path)
    if pc is not None and len(pc) > 0:
        return _mm_filter(pc)

    # 3. float64（MMFi _filtered.bin 的实际格式）
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size % 5 == 0 and raw64.size > 0:
        mm = raw64.reshape(-1, 5)[:, :3].astype(np.float32)
        return _mm_filter(mm)

    return np.empty((0, 3), np.float32)


# ─────────────────────────────────────────────────────────────────
#  LiDAR / RGB / Depth I/O
# ─────────────────────────────────────────────────────────────────

def _frame_num(p):
    m = re.search(r'frame0*(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else -1


def load_lidar_mmfi(path):
    """float64 N×3 = (dep, lat, hgt) 物理坐标。"""
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, 3), np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)


def load_rgb(path):
    if path and os.path.isfile(path):
        if _HAS_PIL:
            return np.array(_PIL.open(path).convert('RGB'))
        arr = plt.imread(path)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr[..., :3]
    return np.full((480, 640, 3), 240, np.uint8)


def load_depth(path):
    """MIST depth: float32 (H,W) 单位米。"""
    if path and os.path.isfile(path):
        return np.load(path).astype(np.float32)
    return np.zeros((480, 640), np.float32)


# ─────────────────────────────────────────────────────────────────
#  归一化（仅用于推理）
# ─────────────────────────────────────────────────────────────────

def _normalize(pc):
    c    = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s    = max(float(np.max(np.sqrt((pc_c ** 2).sum(1)))), 1e-8)
    return (pc_c / s).astype(np.float32), c.astype(np.float32), float(s)


def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0:
        return np.zeros((n, 3), np.float32)
    return pc[np.random.choice(N, n, replace=(N < n))]


# ─────────────────────────────────────────────────────────────────
#  2D 投影（MIST RGB 骨骼叠加）
# ─────────────────────────────────────────────────────────────────

def project_2d(gt_lat_hgt_dep, fx=FX, fy=FY, cx=CX, cy=CY):
    """(lat, hgt_up, dep) → pixel (u, v)"""
    lat, hgt, dep = gt_lat_hgt_dep[:, 0], gt_lat_hgt_dep[:, 1], gt_lat_hgt_dep[:, 2]
    valid = dep > 0.05
    u = np.full(17, np.nan, np.float32)
    v = np.full(17, np.nan, np.float32)
    u[valid] = fx * lat[valid] / dep[valid] + cx
    v[valid] = fy * (-hgt[valid]) / dep[valid] + cy
    return u, v


# ─────────────────────────────────────────────────────────────────
#  序列加载
# ─────────────────────────────────────────────────────────────────

def load_sequence(mmfi_dir, mist_dir,
                  mmfi_start=0, mmfi_end=80,
                  mist_start=0, mist_end=None,
                  mmwave_n=256):
    """
    帧范围对齐加载：将 MMFi[mmfi_start:mmfi_end] 与 MIST[mist_start:mist_end]
    线性拉伸到相同帧数，输出帧由 MMFi 驱动（推理一帧 MMFi 对应一次推理）。

    对第 i 个 MMFi 帧（0-based 相对索引），映射到 MIST 帧：
        mist_fn = mist_start + round(i * (mist_count - 1) / (mmfi_count - 1))

    每帧返回：
      mm_tensor   : (1,N,3) 归一化，推理用
      skel_tensor : (1,17,3) 归一化，推理用
      li_shift    : (1,3) 反归一化用
      li_scale    : float  反归一化用
      mmwave_vis  : (M,3)  物理 (dep,lat,hgt)，mmWave 面板
      lidar_vis   : (K,3)  物理 (dep,lat,hgt)，LiDAR 面板
      skel3d_vis  : (17,3) 物理 (dep,lat,hgt)，骨骼面板
      gen_vis     : None → 推理后填入物理 (dep,lat,hgt)
      rgb         : (H,W,3)
      depth       : (H,W) float32 米
      skel_2d_u/v : (17,) 像素
      n_mm_pts    : int
      frame_num   : int (MMFi 帧索引)
      mist_frame  : int (对应 MIST 帧索引)
    """
    mm_files = sorted(glob.glob(os.path.join(mmfi_dir, 'mmwave', 'frame*_filtered.bin')))
    li_files = sorted(glob.glob(os.path.join(mmfi_dir, 'lidar',  'frame*_filtered.bin')))
    mm_map   = {_frame_num(f): f for f in mm_files}
    li_map   = {_frame_num(f): f for f in li_files}

    mmfi_gt = np.load(os.path.join(mmfi_dir, 'ground_truth.npy')).astype(np.float32)
    T_mmfi  = mmfi_gt.shape[0]

    rgb_files   = sorted(glob.glob(os.path.join(mist_dir, 'rgb',   'frame*.png')))
    depth_files = sorted(glob.glob(os.path.join(mist_dir, 'depth', 'frame*.npy')))
    rgb_map     = {_frame_num(f): f for f in rgb_files}
    depth_map   = {_frame_num(f): f for f in depth_files}
    gt_path = os.path.join(mist_dir, 'ground_truth_h36m.npy')
    if os.path.isfile(gt_path):
        mist_gt = np.load(gt_path).astype(np.float32)
    else:
        print(f'  [警告] 未找到 MIST GT: {gt_path}，2D 骨骼将留空')
        mist_gt = None
    T_mist = len(rgb_files)

    # MMFi 可用帧（仅受 MMFi 数据约束，MIST 单独映射）
    common_mm = sorted(k - 1 for k in mm_map if k >= 1)
    common_li = sorted(k - 1 for k in li_map if k >= 1)
    usable    = sorted(set(common_mm) & set(common_li))
    usable    = [fn for fn in usable if fn < T_mmfi][mmfi_start:mmfi_end]
    if not usable:
        raise RuntimeError('没有可用帧，检查路径和索引范围')

    mmfi_count = len(usable)
    mist_count = (mist_end - mist_start) if mist_end is not None else (T_mist - mist_start)
    mist_count = max(mist_count, 1)

    print(f'  MMFi  : {len(mm_files)} mmwave, {len(li_files)} lidar, GT={T_mmfi}f')
    print(f'  MIST  : {T_mist} RGB, {len(depth_files)} depth')
    print(f'  MMFi 帧范围 [{usable[0]}, {usable[-1]}]，共 {mmfi_count} 帧')
    print(f'  MIST 帧范围 [{mist_start}, {mist_start + mist_count - 1}]，共 {mist_count} 帧')
    print(f'  线性拉伸对齐，输出 {mmfi_count} 帧（由 MMFi 驱动）')

    frames = []
    for i, fn in enumerate(usable):
        # ── 线性插值：将 MIST[mist_count] 拉伸到 MMFi[mmfi_count] ──
        if mmfi_count > 1:
            mist_fn = mist_start + round(i * (mist_count - 1) / (mmfi_count - 1))
        else:
            mist_fn = mist_start

        # ── mmWave：float64 (dep,lat,hgt) 物理坐标 ───────────────
        mm_raw = load_mmwave_mmfi(mm_map[fn + 1])
        if len(mm_raw) == 0:
            mm_raw = np.zeros((1, 3), np.float32)
        # 推理 tensor：独立归一化，轴序不变（与训练 MMFiViPC.py 一致）
        mm_norm, _, _ = _normalize(mm_raw)
        mm_tensor = torch.from_numpy(_subsample(mm_norm, mmwave_n)).float().unsqueeze(0)
        # 显示副本：col2 取反使 Z 向上为正，与 LiDAR 显示方向一致
        mmwave_vis       = mm_raw.copy()
        mmwave_vis[:, 2] *= -1

        # ── LiDAR：float64 (dep,lat,hgt) 物理坐标 ────────────────
        li_xyz = load_lidar_mmfi(li_map[fn + 1])
        if len(li_xyz) < 5:
            li_xyz = np.zeros((10, 3), np.float32)
        _, li_shift, li_scale = _normalize(li_xyz)
        lidar_vis = li_xyz   # 物理坐标，直接显示

        # ── GT skeleton ─────────────────────────────────────────────
        # 原始 (lat, cam_Y, dep)，cam_Y 向下为正（相机 Y 轴约定）
        # → [:,[2,0,1]] → (dep, lat, cam_Y)  用于推理（与训练 MMFiViPC.py 一致）
        # → 显示用：对 col2 取反得到 (dep, lat, hgt_up)，hgt_up 向上为正
        skel_gt = mmfi_gt[fn][:, [2, 0, 1]].copy()
        skel_norm   = (skel_gt - li_shift) / li_scale
        skel_tensor = torch.from_numpy(skel_norm.astype(np.float32)).unsqueeze(0)
        # 显示副本：col2 取反，使 Z 向上为正（与 matplotlib 3D 默认轴向一致）
        skel3d_vis       = skel_gt.copy()
        skel3d_vis[:, 2] *= -1

        # ── MIST RGB + Depth + 2D 投影（使用 mist_fn 而非 fn）────
        rgb_img   = load_rgb(rgb_map.get(mist_fn, None))
        depth_img = load_depth(depth_map.get(mist_fn, None))
        if mist_gt is not None and mist_fn < len(mist_gt):
            u2d, v2d = project_2d(mist_gt[mist_fn])
        else:
            u2d = np.full(17, np.nan, np.float32)
            v2d = np.full(17, np.nan, np.float32)

        frames.append({
            'mm_tensor':   mm_tensor,
            'skel_tensor': skel_tensor,
            'li_shift':    li_shift,
            'li_scale':    li_scale,
            'mmwave_vis':  mmwave_vis,
            'lidar_vis':   lidar_vis,
            'skel3d_vis':  skel3d_vis,
            'gen_vis':     None,
            'rgb':         rgb_img,
            'depth':       depth_img,
            'skel_2d_u':   u2d,
            'skel_2d_v':   v2d,
            'n_mm_pts':    len(mm_raw),
            'frame_num':   fn,
            'mist_frame':  mist_fn,
        })

    mm_pts = [f['n_mm_pts'] for f in frames]
    print(f'  mmWave pts/frame: mean={np.mean(mm_pts):.1f}  '
          f'min={min(mm_pts)}  max={max(mm_pts)}')
    return frames


# ─────────────────────────────────────────────────────────────────
#  模型加载
# ─────────────────────────────────────────────────────────────────

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
    if hasattr(trainer.optimizer, 'swap_parameters_with_ema'):
        trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    trainer.model.eval()
    trainer.compressor.eval()
    print(f'  [Score] epoch={ckpt.get("epoch","?")}')
    ae_cfg = getattr(cfg, 'geo_ae', None)
    geo_ae = GeoAE(
        cfg        = ae_cfg,
        feat_dim   = getattr(ae_cfg, 'feat_dim',   256) if ae_cfg else 256,
        num_coarse = getattr(ae_cfg, 'num_coarse', 128) if ae_cfg else 128,
    ).to(device)
    if ae_ckpt_path and os.path.exists(ae_ckpt_path):
        geo_ae.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
    geo_ae.eval()
    return trainer, geo_ae


# ─────────────────────────────────────────────────────────────────
#  推理（反归一化到物理坐标）
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_sequence(trainer, geo_ae, frames, device):
    label   = torch.tensor([ACTION_LABEL], dtype=torch.long).to(device)
    t_total = 0.
    for i, fr in enumerate(frames):
        mm   = fr['mm_tensor'].to(device)
        skel = fr['skel_tensor'].to(device)
        coarse, _ = geo_ae(mm)
        t0 = time.time()
        pts, _ = trainer.sample(1, label=label,
                                condition={'skeleton': skel, 'coarse': coarse})
        t_total += time.time() - t0
        gen_norm = pts[0].cpu().numpy()
        fr['gen_vis'] = (gen_norm * fr['li_scale'] + fr['li_shift']).astype(np.float32)
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f'  推理 {i+1}/{len(frames)}  {t_total/(i+1)*1000:.0f} ms/帧')
    return frames


# ─────────────────────────────────────────────────────────────────
#  绘图工具
# ─────────────────────────────────────────────────────────────────

def _compute_axlims(arrays, pad=0.15):
    """从点云列表计算固定等比例立方体轴范围。"""
    all_pts = np.vstack([a for a in arrays if len(a) > 0])
    lo = np.percentile(all_pts, 2,  axis=0)
    hi = np.percentile(all_pts, 98, axis=0)
    mid  = (lo + hi) / 2
    span = max((hi - lo).max() / 2 + pad, 0.4)
    return ((mid[0]-span, mid[0]+span),
            (mid[1]-span, mid[1]+span),
            (mid[2]-span, mid[2]+span))


def _style3d(ax, title, axlims, elev, azim):
    ax.set_facecolor('white')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = True
        p.set_facecolor('#F5F5F5')
        p.set_edgecolor('#CCCCCC')
    ax.tick_params(colors='#555', labelsize=8)
    ax.grid(True, color='#E0E0E0', linewidth=0.4)
    ax.set_title(title, color='black', fontsize=14, pad=4)
    (x0, x1), (y0, y1), (z0, z1) = axlims
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_xlabel('Depth',   fontsize=9, color='#444', labelpad=1)
    ax.set_ylabel('Lateral', fontsize=9, color='#444', labelpad=1)
    ax.set_zlabel('Height',  fontsize=9, color='#444', labelpad=1)
    ax.view_init(elev=elev, azim=azim)


def _draw_skel3d(ax, js, lw=2.2, alpha=0.9):
    for (a, b) in BONES:
        col = _BC.get((a, b), _BC.get((b, a), '#888'))
        ax.plot([js[a,0],js[b,0]], [js[a,1],js[b,1]], [js[a,2],js[b,2]],
                color=col, lw=lw, alpha=alpha)
    ax.scatter(js[:,0], js[:,1], js[:,2],
               c='#222', s=20, zorder=5, edgecolors='white', lw=0.5)


CROP_X_RGB   =  70  # RGB 左右各裁剪像素（直接裁剪数组，无拉伸）
CROP_X_DEPTH =  70  # 深度图左右各裁剪像素

def _draw_skel2d(ax, rgb, u, v, img_hw):
    H, W = img_hw
    # 裁剪数组本身，避免 aspect='auto' 拉伸
    rgb_c = rgb[:, CROP_X_RGB: W - CROP_X_RGB]
    Wc = rgb_c.shape[1]
    ax.imshow(rgb_c, extent=[0, Wc, H, 0], aspect='auto')
    ax.set_xlim(0, Wc); ax.set_ylim(H, 0); ax.axis('off')
    # 骨骼坐标随裁剪平移
    us = u - CROP_X_RGB
    for (a, b) in BONES:
        if np.isfinite(us[a]) and np.isfinite(us[b]):
            col = _BC.get((a,b), _BC.get((b,a), '#888'))
            ax.plot([us[a], us[b]], [v[a], v[b]], color=col, lw=2, alpha=0.9)
    ok = np.isfinite(us) & np.isfinite(v)
    if ok.any():
        ax.scatter(us[ok], v[ok], c='white', s=18, zorder=5,
                   edgecolors='#333', lw=0.5)


def _draw_depth(ax, depth_m, title='Depth'):
    masked = np.where((depth_m > 0.1) & (depth_m < 6.0), depth_m, np.nan)
    # 裁剪数组本身
    masked_c = masked[:, CROP_X_DEPTH: masked.shape[1] - CROP_X_DEPTH]
    im = ax.imshow(masked_c, cmap='plasma_r', vmin=1.0, vmax=4.5,
                   aspect='auto', interpolation='nearest')
    ax.axis('off')
    ax.set_title(title, color='black', fontsize=14, pad=4)
    ax.set_facecolor('white')
    return im


# ─────────────────────────────────────────────────────────────────
#  动画生成（2×3 布局）
# ─────────────────────────────────────────────────────────────────

def make_animation(frames, save_path, fps=8, elev=20, azim=-55):
    """
    2×3 布局（白底，物理坐标，固定轴）：
      行0: RGB+2D骨骼 | 深度图    | LiDAR GT
      行1: mmWave      | 3D 骨骼  | PRISM 生成
    """
    # LiDAR/PRISM 共用轴范围（物理坐标，未翻转 Z）
    axlims_li = _compute_axlims([fr['lidar_vis'] for fr in frames])
    # skeleton/mmwave 用翻转后的 Z 计算自己的轴范围
    axlims_sk = _compute_axlims([fr['skel3d_vis'] for fr in frames])
    axlims_mm = _compute_axlims([fr['mmwave_vis']  for fr in frames if len(fr['mmwave_vis']) > 1])
    print(f'  轴范围 LiDAR: dep={axlims_li[0]}, lat={axlims_li[1]}, hgt={axlims_li[2]}')
    print(f'  轴范围 Skel : dep={axlims_sk[0]}, lat={axlims_sk[1]}, hgt={axlims_sk[2]}')
    print(f'  轴范围 mmWave: dep={axlims_mm[0]}, lat={axlims_mm[1]}, hgt={axlims_mm[2]}')

    fig = plt.figure(figsize=(20, 9), facecolor='white')
    gs  = GridSpec(2, 3, figure=fig,
                   hspace=0.10, wspace=0.05,
                   left=0.01, right=0.99,
                   top=0.97, bottom=0.04)

    ax_rgb   = fig.add_subplot(gs[0, 0])
    ax_depth = fig.add_subplot(gs[0, 1])
    ax_li    = fig.add_subplot(gs[0, 2], projection='3d')
    ax_mm    = fig.add_subplot(gs[1, 0], projection='3d')
    ax_sk    = fig.add_subplot(gs[1, 1], projection='3d')
    ax_gen   = fig.add_subplot(gs[1, 2], projection='3d')

    # colormap Z 范围：LiDAR 面板和 PRISM 面板用 axlims_li 的 Z
    z_lo_li, z_hi_li = axlims_li[2]
    z_lo_mm, z_hi_mm = axlims_mm[2]

    def update(i):
        fr = frames[i]

        # ── (0,0) RGB + MIST 2D 骨骼 ───────────────────────────
        ax_rgb.cla()
        ax_rgb.set_facecolor('white')
        ax_rgb.set_title('RGB + H36M-17', color='black', fontsize=14, pad=4)
        H, W = fr['rgb'].shape[:2]
        _draw_skel2d(ax_rgb, fr['rgb'], fr['skel_2d_u'], fr['skel_2d_v'], (H, W))

        # ── (0,1) 深度图 ────────────────────────────────────────
        ax_depth.cla()
        _draw_depth(ax_depth, fr['depth'])

        # ── (0,2) LiDAR GT 点云（Z 未翻转，LiDAR 传感器坐标系）──
        ax_li.cla()
        _style3d(ax_li, 'LiDAR GT', axlims_li, elev, azim)
        li = fr['lidar_vis']
        if len(li) > 0:
            ax_li.scatter(li[:,0], li[:,1], li[:,2],
                          c=li[:,2], cmap='turbo', s=5,
                          vmin=z_lo_li, vmax=z_hi_li,
                          depthshade=False, lw=0, alpha=0.85)

        # ── (1,0) mmWave（col2 已翻转为 hgt_up，自己的轴范围）──
        ax_mm.cla()
        _style3d(ax_mm, 'mmWave', axlims_mm, elev, azim)
        mm = fr['mmwave_vis']
        if len(mm) > 0:
            ax_mm.scatter(mm[:,0], mm[:,1], mm[:,2],
                          c=mm[:,2], cmap='autumn_r', s=32, alpha=0.9,
                          vmin=z_lo_mm, vmax=z_hi_mm,
                          depthshade=False, lw=0)

        # ── (1,1) 3D 骨骼（col2 已翻转为 hgt_up，独立轴范围）──
        ax_sk.cla()
        _style3d(ax_sk, 'GT Skeleton', axlims_sk, elev, azim)
        _draw_skel3d(ax_sk, fr['skel3d_vis'], lw=2.5, alpha=0.95)

        # ── (1,2) PRISM 生成（反归一化到 LiDAR 坐标系）─────────
        ax_gen.cla()
        _style3d(ax_gen, 'PRISM Output', axlims_li, elev, azim)
        gen = fr['gen_vis']
        if gen is not None and len(gen) > 0:
            ax_gen.scatter(gen[:,0], gen[:,1], gen[:,2],
                           c=gen[:,2], cmap='cool', s=6,
                           vmin=z_lo_li, vmax=z_hi_li,
                           depthshade=False, lw=0, alpha=0.9)

        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    ani.save(save_path,
             writer=animation.FFMpegWriter(fps=fps, bitrate=5000), dpi=100)
    plt.close(fig)
    print(f'  已保存: {save_path}  ({len(frames)}帧, {len(frames)/fps:.1f}s)')


# ─────────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser('PRISM Demo 视频生成器')
    pa.add_argument('--mmfi_dir',   default='data/MMFi/E01/S06/A27')
    pa.add_argument('--mist_dir',   default='assets/DataCollection/collected_data/jiacheng_jugong')
    pa.add_argument('--cfg',        default='experiments/Latent_Diffusion_Trainer/config_8actions.yaml')
    pa.add_argument('--ckpt',       default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_3600.pth')
    pa.add_argument('--ae',         default='experiments/Latent_Diffusion_Trainer/mmfi_202603151358/autoencoder_e_3600.pth')
    pa.add_argument('--out',        default='outputs/demo')
    # ── 帧范围（两侧独立指定，线性拉伸对齐）─────────────────────────
    pa.add_argument('--mmfi_start', type=int,   default=0,
                    help='MMFi 起始帧索引（0-based，含）')
    pa.add_argument('--mmfi_end',   type=int,   default=80,
                    help='MMFi 结束帧索引（exclusive）')
    pa.add_argument('--mist_start', type=int,   default=0,
                    help='MIST 起始帧索引（0-based，含）')
    pa.add_argument('--mist_end',   type=int,   default=None,
                    help='MIST 结束帧索引（exclusive，None=到末尾）')
    # ── 输出控制 ──────────────────────────────────────────────────
    pa.add_argument('--fps',        type=int,   default=5,
                    help='输出视频帧率（默认 5，较慢便于观察）')
    pa.add_argument('--fmt',        default='mp4')
    pa.add_argument('--elev',       type=float, default=20)
    pa.add_argument('--azim',       type=float, default=-55)
    pa.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = pa.parse_args()

    out_fps = args.fps
    print(f'Device: {args.device}')
    print(f'MMFi 帧 [{args.mmfi_start}, {args.mmfi_end})  '
          f'MIST 帧 [{args.mist_start}, {args.mist_end or "末尾"})  '
          f'输出 {out_fps}fps\n')

    device = torch.device(args.device)

    print('[1/3] 加载数据...')
    frames = load_sequence(
        args.mmfi_dir, args.mist_dir,
        mmfi_start=args.mmfi_start, mmfi_end=args.mmfi_end,
        mist_start=args.mist_start, mist_end=args.mist_end,
    )

    print('\n[2/3] 加载模型...')
    trainer, geo_ae = load_models(args.cfg, args.ckpt, args.ae, device)

    print('\n[3/3] 推理...')
    frames = infer_sequence(trainer, geo_ae, frames, device)

    fname     = f'demo_S06A27_f{args.mmfi_start}-{args.mmfi_end}.{args.fmt}'
    save_path = os.path.join(args.out, fname)
    print(f'\n[保存] {save_path}')
    make_animation(frames, save_path, fps=out_fps, elev=args.elev, azim=args.azim)
    print('完成。')


if __name__ == '__main__':
    main()
