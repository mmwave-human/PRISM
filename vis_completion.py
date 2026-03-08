"""
vis_completion.py
=================
点云补全三列对比可视化：Input (mmWave) | Generated | GT (LiDAR)

坐标轴约定（与 MMFi LiDAR/mmWave 一致）：
  index 0 → X : 深度方向 (depth)
  index 1 → Y : 左右方向 (lateral)
  index 2 → Z : 高度方向 (height, 向上为正)

特性：
  - 三列共享同一坐标轴范围（以 GT 为基准），消除视觉偏差
  - Input 用更大点 + 高亮色，解决 mmWave 稀疏看不清的问题
  - Z 轴着色（高度渐变），帮助判断人体姿态上下关系
  - 同时输出 3D 视图 + XZ 正视图 + YZ 侧视图，多角度对比
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os


# ─────────────────────────────────────────────
#  数据加载
# ─────────────────────────────────────────────

def load_npy(path):
    data = np.load(path)
    print(f"Loaded {path}: shape={data.shape}, "
          f"xyz range: x=[{data[...,0].min():.2f},{data[...,0].max():.2f}] "
          f"y=[{data[...,1].min():.2f},{data[...,1].max():.2f}] "
          f"z=[{data[...,2].min():.2f},{data[...,2].max():.2f}]")
    return data


# ─────────────────────────────────────────────
#  坐标轴范围（以 GT 为基准，三列共用）
# ─────────────────────────────────────────────

def compute_shared_limits(ref_batch, padding=0.15):
    """
    以 GT 点云的百分位数计算坐标范围，加 padding 留白。
    返回 (xlim, ylim, zlim)，每个是 (lo, hi) 元组。
    """
    pts = ref_batch.reshape(-1, 3)
    def pct_lim(col):
        lo = float(np.percentile(pts[:, col], 1))
        hi = float(np.percentile(pts[:, col], 99))
        span = max(hi - lo, 0.3)
        pad  = span * padding
        return lo - pad, hi + pad

    return pct_lim(0), pct_lim(1), pct_lim(2)


# ─────────────────────────────────────────────
#  单个 3D 子图绘制
# ─────────────────────────────────────────────

def draw_3d(ax, points, title, xlim, ylim, zlim,
            point_size=8, cmap='RdYlGn', alpha=0.85,
            elev=20, azim=-60):
    """
    在 ax 上绘制单个点云，Z 轴高度着色。
    points: (N, 3)，坐标约定 (depth, lateral, height)
    """
    if points.ndim == 3:
        points = points[0]

    ax.clear()
    ax.set_facecolor('#0F0F1A')

    # 去掉坐标面填充，让点云更清晰
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#222244')
    ax.yaxis.pane.set_edgecolor('#222244')
    ax.zaxis.pane.set_edgecolor('#222244')
    ax.tick_params(colors='gray', labelsize=6)
    ax.grid(True, color='#1E1E3A', linewidth=0.4)

    if len(points) > 0:
        # Z 轴着色，clip 到当前范围内防止 colormap 失真
        z_vals = np.clip(points[:, 2], zlim[0], zlim[1])
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=z_vals, cmap=cmap,
            s=point_size, alpha=alpha,
            depthshade=False, linewidths=0
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel('X(depth)',   color='gray', fontsize=6, labelpad=2)
    ax.set_ylabel('Y(lateral)', color='gray', fontsize=6, labelpad=2)
    ax.set_zlabel('Z(height)',  color='gray', fontsize=6, labelpad=2)
    ax.set_title(title, color='white', fontsize=9, pad=4)
    ax.view_init(elev=elev, azim=azim)


# ─────────────────────────────────────────────
#  单个 2D 投影子图绘制
# ─────────────────────────────────────────────

def draw_2d(ax, points, xi, yi, xlabel, ylabel,
            xlim, ylim, point_size=8, cmap='RdYlGn', alpha=0.85):
    """
    2D 投影（XZ 正视 或 YZ 侧视），颜色仍用 Z 高度。
    """
    ax.clear()
    ax.set_facecolor('#0F0F1A')
    ax.tick_params(colors='gray', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color('#222244')
    ax.grid(True, color='#1E1E3A', linewidth=0.4)

    if len(points) > 0:
        z_vals = points[:, 2]  # 始终用 Z 着色
        ax.scatter(
            points[:, xi], points[:, yi],
            c=z_vals, cmap=cmap,
            s=point_size, alpha=alpha, linewidths=0
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, color='gray', fontsize=7)
    ax.set_ylabel(ylabel, color='gray', fontsize=7)


# ─────────────────────────────────────────────
#  主可视化函数
# ─────────────────────────────────────────────

def visualize_completion(smp_path, ref_path, part_path,
                         num_samples=4, save_path=None,
                         views=('3d', 'xz', 'yz'),
                         elev=20, azim=-60):
    """
    三列对比：Input (mmWave) | Generated | GT (LiDAR)
    每列支持多视角行：3D、XZ 正视、YZ 侧视

    views: tuple，指定要显示哪些视角，顺序即行顺序
           可选 '3d', 'xz', 'yz'
    """
    smp  = load_npy(smp_path)
    ref  = load_npy(ref_path)
    part = load_npy(part_path)

    n = min(num_samples, smp.shape[0], ref.shape[0], part.shape[0])
    smp  = smp[:n]
    ref  = ref[:n]
    part = part[:n]

    # 共享坐标轴（以 GT 为基准）
    xlim, ylim, zlim = compute_shared_limits(ref)

    num_views = len(views)
    # 列数 = 样本数 × 3（input / generated / gt）
    num_cols = n * 3

    fig = plt.figure(figsize=(4.5 * num_cols / 3, 4.0 * num_views + 0.5),
                     facecolor='#0F0F1A')
    fig.suptitle('Point Cloud Completion Visualization',
                 color='white', fontsize=14, y=1.01)

    col_labels = []
    for i in range(n):
        col_labels += [f'Input (mmWave)\nSample {i+1}',
                       f'Generated\nSample {i+1}',
                       f'GT (LiDAR)\nSample {i+1}']

    # 每个样本的三列点云和对应显示参数
    # Input 用更大点 + 不同 cmap 突出稀疏点
    datasets = []
    for i in range(n):
        datasets.append((part[i], 'plasma', 18, 0.95))   # Input：大点，plasma 橙紫色
        datasets.append((smp[i],  'RdYlGn', 6,  0.80))   # Generated：中点，绿红
        datasets.append((ref[i],  'RdYlGn', 4,  0.75))   # GT：小点，绿红

    for row_idx, view in enumerate(views):
        for col_idx, (pts, cmap, psize, alpha) in enumerate(datasets):
            subplot_idx = row_idx * num_cols + col_idx + 1

            if view == '3d':
                ax = fig.add_subplot(num_views, num_cols, subplot_idx,
                                     projection='3d')
                title = col_labels[col_idx] if row_idx == 0 else ''
                draw_3d(ax, pts, title, xlim, ylim, zlim,
                        point_size=psize, cmap=cmap, alpha=alpha,
                        elev=elev, azim=azim)

            elif view == 'xz':
                ax = fig.add_subplot(num_views, num_cols, subplot_idx)
                title = col_labels[col_idx] if row_idx == 0 else ''
                ax.set_title(title, color='white', fontsize=9, pad=4)
                draw_2d(ax, pts, xi=0, yi=2,
                        xlabel='X(depth)', ylabel='Z(height)',
                        xlim=xlim, ylim=zlim,
                        point_size=psize, cmap=cmap, alpha=alpha)

            elif view == 'yz':
                ax = fig.add_subplot(num_views, num_cols, subplot_idx)
                title = col_labels[col_idx] if row_idx == 0 else ''
                ax.set_title(title, color='white', fontsize=9, pad=4)
                draw_2d(ax, pts, xi=1, yi=2,
                        xlabel='Y(lateral)', ylabel='Z(height)',
                        xlim=ylim, ylim=zlim,
                        point_size=psize, cmap=cmap, alpha=alpha)

    # 行标签：视角说明
    view_labels = {'3d': '3D View', 'xz': 'XZ Front View\n(depth vs height)',
                   'yz': 'YZ Side View\n(lateral vs height)'}
    for row_idx, view in enumerate(views):
        fig.text(0.01,
                 1.0 - (row_idx + 0.5) / num_views,
                 view_labels.get(view, view),
                 color='#AAAACC', fontsize=8, va='center',
                 rotation=90)

    plt.tight_layout(rect=[0.03, 0, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0F0F1A')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser('Point Cloud Completion Visualization')
    parser.add_argument('--smp',   type=str, required=True,
                        help='生成结果 npy 路径')
    parser.add_argument('--ref',   type=str, required=True,
                        help='GT 参考点云 npy 路径')
    parser.add_argument('--part',  type=str, required=True,
                        help='残缺输入点云 npy 路径')
    parser.add_argument('--num',   type=int, default=4,
                        help='显示样本数量（默认 4）')
    parser.add_argument('--save',  type=str, default=None,
                        help='保存图片路径，不指定则弹窗显示')
    parser.add_argument('--views', type=str, default='3d,xz,yz',
                        help='显示的视角，逗号分隔，可选 3d/xz/yz（默认 3d,xz,yz）')
    parser.add_argument('--elev',  type=float, default=20,
                        help='3D 视角仰角（默认 20）')
    parser.add_argument('--azim',  type=float, default=-60,
                        help='3D 视角方位角（默认 -60）')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    views = [v.strip() for v in args.views.split(',')]
    visualize_completion(
        smp_path=args.smp,
        ref_path=args.ref,
        part_path=args.part,
        num_samples=args.num,
        save_path=args.save,
        views=views,
        elev=args.elev,
        azim=args.azim,
    )