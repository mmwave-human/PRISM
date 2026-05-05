"""
vis_occlusion_levels.py
=======================
使用 MMFi mmWave 真实数据生成三种稀疏度等级的正面点云对比图。

策略：
  ① 加载并聚合不同帧数的真实 mmWave 数据
  ② 以 Dense 云为人体参考模板，对稀疏级别补充含噪声的合成点
     → 大致符合人体分布，但保留毫米波雷达的抽象感
  ③ 各级控制目标点数，符合 OL2/OL1/OL0 的典型范围

用法：
    python vis_occlusion_levels.py \
        --action A13 --center 80 \
        --out assets/fig/occlusion_levels.png --dpi 200
"""

import argparse
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── 默认参数 ─────────────────────────────────────────────────
DATA_ROOT = 'data/MMFi'
ENV       = 'E01'
SUBJECT   = 'S01'

# 三级聚合帧数 (Sparse / Medium / Dense)
DEFAULT_N_FRAMES  = [1, 6, 20]
# 目标点数，对应 OL2 / OL1 / OL0 典型范围
DEFAULT_TARGET_N  = [20, 55, 120]
# 各级真实数据占比（其余从 Dense 模板补充含噪合成点）
DEFAULT_REAL_FRAC = [0.55, 0.70, 0.90]
# 合成点高斯噪声 σ（归一化坐标），越大越抽象
DEFAULT_SIGMA     = 0.055

LEVEL_LABELS = ['Sparse', 'Medium', 'Dense']
POINT_COLOR  = '#4D96FF'


# ── mmWave 加载（与 vis_demo.py 一致）───────────────────────
def _mm_filter(pc):
    keep = (
        (pc[:, 0] > 0.5) &
        (np.abs(pc[:, 1]) < 3.0) &
        (pc[:, 2] > -2.0) &
        (pc[:, 2] < 2.5)
    )
    return pc[keep]


def _load_mmwave(path):
    """float64 优先（MMFi _filtered.bin 实际格式），返回 (N,3) 物理坐标。"""
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size % 5 == 0 and raw64.size > 0:
        xyz = raw64.reshape(-1, 5)[:, :3].astype(np.float32)
        f = _mm_filter(xyz)
        if len(f) > 0:
            return f
    raw32 = np.fromfile(path, dtype=np.float32)
    if raw32.size % 5 == 0 and raw32.size > 0:
        pc = raw32.reshape(-1, 5)[:, :3]
        if np.all(np.abs(pc) < 20):
            return _mm_filter(pc)
    return np.empty((0, 3), dtype=np.float32)


def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1


def _collect_raw(mmwave_dir, center_frame, n_frames):
    """聚合 n_frames 帧，返回原始物理坐标点云。"""
    bins = sorted(glob.glob(os.path.join(mmwave_dir, '*.bin')), key=_frame_num)
    nums = [_frame_num(b) for b in bins]
    ci   = int(np.argmin([abs(n - center_frame) for n in nums]))
    half = n_frames // 2
    lo   = max(0, ci - half)
    hi   = min(len(bins) - 1, ci + (n_frames - half) - 1)
    pts  = [_load_mmwave(b) for b in bins[lo: hi + 1]]
    pts  = [p for p in pts if len(p) > 0]
    return np.concatenate(pts, axis=0) if pts else np.zeros((1, 3), np.float32)


# ── 核心：点云增强 ────────────────────────────────────────────
def _subsample(pc, n, rng):
    if len(pc) <= n:
        return pc
    return pc[rng.choice(len(pc), n, replace=False)]


def _augment(real_norm, ref_norm, target_n, real_frac, sigma, rng):
    """
    混合真实点 + 从 Dense 模板采样的含噪合成点，达到 target_n 个点。

    real_norm : 当前等级归一化点云
    ref_norm  : Dense 归一化点云（人体模板）
    sigma     : 合成点噪声 σ（在归一化坐标中）
    """
    n_real  = min(len(real_norm), max(1, int(target_n * real_frac)))
    n_synth = max(0, target_n - n_real)

    # 真实点（下采样）
    pts_real = _subsample(real_norm, n_real, rng)

    # 合成点：从 Dense 模板随机采，加各向异性高斯噪声
    if n_synth > 0 and len(ref_norm) > 0:
        idx     = rng.choice(len(ref_norm), n_synth, replace=True)
        centers = ref_norm[idx]
        # 深度方向噪声稍大（雷达深度精度低于横向），高度方向适中
        noise   = rng.standard_normal(centers.shape).astype(np.float32)
        noise[:, 0] *= sigma * 1.4   # dep
        noise[:, 1] *= sigma         # lat
        noise[:, 2] *= sigma * 0.9   # hgt
        pts_synth = centers + noise
        return np.concatenate([pts_real, pts_synth], axis=0)

    return pts_real


def build_levels(mmwave_dir, center_frame,
                 n_frames_list, target_n_list, real_frac_list,
                 sigma=DEFAULT_SIGMA, seed=42):
    """
    构建三级点云（归一化坐标）。
    以 Dense 原始帧云为归一化基准 & 人体模板。
    """
    rng = np.random.default_rng(seed)

    # 公共归一化参数（用 Dense 帧云）
    ref_raw = _collect_raw(mmwave_dir, center_frame, max(n_frames_list))
    shift   = ref_raw.mean(axis=0)
    ref_c   = ref_raw - shift
    scale   = max(float(np.max(np.linalg.norm(ref_c, axis=1))), 1e-8)

    # Dense 模板（归一化 + 下采样到 Dense 目标点数）
    ref_norm = (ref_c / scale).astype(np.float32)
    ref_tmpl = _subsample(ref_norm, max(target_n_list), rng)

    result = []
    for nf, target_n, frac in zip(n_frames_list, target_n_list, real_frac_list):
        raw  = _collect_raw(mmwave_dir, center_frame, nf)
        norm = ((raw - shift) / scale).astype(np.float32)
        pc   = _augment(norm, ref_tmpl, target_n, frac, sigma, rng)
        result.append(pc)
        print(f'  n_frames={nf:2d}  real={len(norm):3d}  '
              f'→ final={len(pc):3d} pts  (target={target_n})')
    return result


# ── 自定义 colormap（与论文图色调一致）────────────────────────
from matplotlib.colors import LinearSegmentedColormap

# 深紫（边框色 #6040A8）→ 品红 → 橙（Daily Living #E07834）→ 亮黄
_OC_COLORS = ['#160840', '#6040A8', '#C03880', '#E07834', '#FFD000']
OC_CMAP = LinearSegmentedColormap.from_list('oc_prism', _OC_COLORS, N=256)


# ── 渲染 ─────────────────────────────────────────────────────
def render(pc_list, labels, out_path,
           dpi=150, point_size=9, alpha=0.90,
           color=POINT_COLOR, fig_w=13, fig_h=5.6,
           pts_ranges=('50–50–150 pts', '25–80 pts', '10–30 pts'),
           attenuation_label='Signal Attenuation'):
    """
    正面 2D 投影：lateral(Y) x height(Z)。
    方案一：自定义 colormap（深紫→橙→黄），与论文色调一致。
    方案二：内嵌 → 箭头 + 底部渐变衰减条，风格对齐 MIST 论文。
    """
    n = len(pc_list)

    # 顺序：Dense → Medium → Sparse（与衰减箭头方向一致）
    pc_list    = list(pc_list[::-1])
    labels     = list(labels[::-1])
    pts_ranges = list(pts_ranges[::-1])   # 同步翻转

    # 布局：3 点云列 + 2 箭头列；下方衰减条
    # cols: [pc, arrow, pc, arrow, pc]
    width_ratios = [1, 0.13, 1, 0.13, 1]
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs  = GridSpec(2, 5, figure=fig,
                   height_ratios=[1, 0.09],
                   width_ratios=width_ratios,
                   hspace=0.30, wspace=0.0)

    # 坐标范围（Dense 云决定，翻转后第 0 个）
    dense_pc         = pc_list[0]
    lat_d, hgt_d     = dense_pc[:, 1], dense_pc[:, 2]
    pad              = 0.20
    lat_min, lat_max = lat_d.min() - pad, lat_d.max() + pad
    hgt_min, hgt_max = hgt_d.min() - pad, hgt_d.max() + pad

    # 全局高度范围（三图共享色标）
    all_hgt            = np.concatenate([pc[:, 2] for pc in pc_list])
    hgt_vmin, hgt_vmax = all_hgt.min(), all_hgt.max()

    for i, (pc, label, pts_r) in enumerate(zip(pc_list, labels, pts_ranges)):
        col = i * 2   # 0, 2, 4
        ax  = fig.add_subplot(gs[0, col])
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        lat = pc[:, 1]
        hgt = pc[:, 2]

        # 方案一：plasma colormap 按高度着色（与 MIST 原图一致）
        ax.scatter(lat, hgt, s=point_size, c=hgt,
                   cmap='plasma', vmin=hgt_vmin, vmax=hgt_vmax,
                   alpha=alpha, linewidths=0, rasterized=True)

        ax.set_xlim(lat_min, lat_max)
        ax.set_ylim(hgt_min, hgt_max)

        # 标签：OL 名称（粗）+ 点数范围（灰小字），对齐 MIST 风格
        ax.set_title(label, fontsize=13, fontweight='bold',
                     pad=4, color='#111111')
        ax.text(0.5, -0.04, pts_r,
                ha='center', va='top', fontsize=9.5,
                color='#444444', transform=ax.transAxes)

        # 方案二（内嵌）：相邻两列插入 → 箭头（与 MIST 原图一致：简洁黑色）
        if i < n - 1:
            ax_arr = fig.add_subplot(gs[0, col + 1])
            ax_arr.set_axis_off()
            ax_arr.text(0.5, 0.5, u'→',
                        ha='center', va='center', fontsize=20,
                        color='#222222', fontweight='normal',
                        transform=ax_arr.transAxes)

    # 方案二（底部）：渐变衰减条
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_axis_off()

    # 深蓝→浅蓝渐变（左=信号强 → 右=信号弱）
    grad = np.linspace(0.9, 0.08, 256).reshape(1, -1)
    bar_cmap = LinearSegmentedColormap.from_list(
        'bar', ['#3A4FCF', '#8899DD', '#DDE4F5'])
    ax_bar.imshow(grad, aspect='auto', cmap=bar_cmap,
                  extent=[0, 1, 0, 1], alpha=0.65,
                  transform=ax_bar.transAxes)

    ax_bar.annotate('', xy=(0.97, 0.5), xytext=(0.03, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#222244',
                                   lw=2.0, mutation_scale=20))
    ax_bar.text(0.5, -0.6, attenuation_label,
                ha='center', va='top', fontsize=10,
                color='#333355', fontstyle='italic',
                transform=ax_bar.transAxes)

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', transparent=False)
    plt.close(fig)
    print(f'saved -> {out_path}')


# ── 入口 ─────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  default=DATA_ROOT)
    p.add_argument('--env',        default=ENV)
    p.add_argument('--subject',    default=SUBJECT)
    p.add_argument('--action',     default='A13')
    p.add_argument('--center',     type=int, default=80)
    p.add_argument('--n_frames',   type=int, nargs=3, default=DEFAULT_N_FRAMES,
                   metavar=('SP', 'MD', 'DN'))
    p.add_argument('--target_n',   type=int, nargs=3, default=DEFAULT_TARGET_N,
                   metavar=('SP', 'MD', 'DN'),
                   help='三个稀疏级别的目标点数')
    p.add_argument('--real_frac',  type=float, nargs=3, default=DEFAULT_REAL_FRAC,
                   metavar=('SP', 'MD', 'DN'),
                   help='各级真实数据占比，其余从模板补充')
    p.add_argument('--sigma',      type=float, default=DEFAULT_SIGMA,
                   help='合成点高斯噪声 σ，越大越抽象')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--labels',     nargs=3,    default=['OL2', 'OL1', 'OL0'])
    p.add_argument('--point_size', type=float, default=9)
    p.add_argument('--alpha',      type=float, default=0.82)
    p.add_argument('--color',      default=POINT_COLOR)
    p.add_argument('--out',        default='assets/fig/occlusion_levels.png')
    p.add_argument('--dpi',        type=int,   default=150)
    args = p.parse_args()

    mmwave_dir = os.path.join(
        args.data_root, args.env, args.subject, args.action, 'mmwave')
    assert os.path.isdir(mmwave_dir), f'目录不存在: {mmwave_dir}'

    pc_list = build_levels(
        mmwave_dir, args.center,
        args.n_frames, args.target_n, args.real_frac,
        sigma=args.sigma, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    render(pc_list, args.labels, args.out,
           dpi=args.dpi, point_size=args.point_size,
           alpha=args.alpha, color=args.color,
           pts_ranges=('10–30 pts', '25–80 pts', '50–150 pts'))


if __name__ == '__main__':
    main()
