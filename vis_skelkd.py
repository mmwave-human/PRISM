"""
vis_skelkd.py
可视化 SkelKD 预测骨骼 vs GT 骨骼对比
用法：
  python vis_skelkd.py                        # 默认看 epoch 200
  python vis_skelkd.py --epoch 100            # 看指定 epoch
  python vis_skelkd.py --epoch 200 --n 8      # 显示 8 个样本
  python vis_skelkd.py --curve               # 显示 MPJPE 收敛曲线
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 无显示器服务器也能保存图片
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MM-Fi 17关节骨骼连接对（COCO格式）
BONES = [
    (0,1),(0,2),(1,3),(2,4),          # 头部
    (3,5),(4,6),(5,6),                 # 肩
    (5,7),(7,9),(6,8),(8,10),          # 手臂
    (5,11),(6,12),(11,12),             # 躯干
    (11,13),(13,15),(12,14),(14,16),   # 腿
]

SAVE_DIR = 'experiments/SkelKD/mmfi_skelkd'


def draw_skeleton(ax, joints, color, alpha=1.0, label=None):
    """joints: (17, 3)  xyz"""
    for i, (a, b) in enumerate(BONES):
        xs = [joints[a, 0], joints[b, 0]]
        ys = [joints[a, 1], joints[b, 1]]
        zs = [joints[a, 2], joints[b, 2]]
        ax.plot(xs, ys, zs, color=color, alpha=alpha,
                linewidth=1.5, label=label if i == 0 else None)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               color=color, s=15, alpha=alpha, depthshade=False)


def vis_samples(epoch, n_samples=6):
    pred_path = os.path.join(SAVE_DIR, f'pred_joints_ep{epoch}.npy')
    gt_path   = os.path.join(SAVE_DIR, f'gt_joints_ep{epoch}.npy')

    if not os.path.exists(pred_path):
        print(f'[ERROR] 找不到 {pred_path}')
        print('可用的 epoch：', sorted([
            int(f.replace('pred_joints_ep','').replace('.npy',''))
            for f in os.listdir(SAVE_DIR) if f.startswith('pred_joints')
        ]))
        return

    pred = np.load(pred_path)   # (N, 17, 3)
    gt   = np.load(gt_path)     # (N, 17, 3)

    # 计算每个样本的 MPJPE，选最好/中等/最差的样本展示
    mpjpe_per_sample = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)  # (N,)
    n_samples = min(n_samples, len(pred))
    sorted_idx = np.argsort(mpjpe_per_sample)
    # 均匀取样：最好3个 + 中间 + 最差2个
    step = max(len(sorted_idx) // n_samples, 1)
    indices = sorted_idx[::step][:n_samples]

    fig = plt.figure(figsize=(n_samples * 3.5, 7))
    fig.suptitle(f'SkelKD  Epoch {epoch}  |  蓝=预测  橙=GT\n'
                 f'整体 MPJPE={mpjpe_per_sample.mean():.4f} (归一化单位)',
                 fontsize=13)

    for col, idx in enumerate(indices):
        # ── 正视图（XZ平面，depth-height）──
        ax1 = fig.add_subplot(2, n_samples, col + 1, projection='3d')
        draw_skeleton(ax1, pred[idx], color='royalblue', label='Pred')
        draw_skeleton(ax1, gt[idx],   color='darkorange', label='GT')
        ax1.set_title(f'#{idx}\nMPJPE={mpjpe_per_sample[idx]:.4f}', fontsize=8)
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        ax1.view_init(elev=15, azim=-70)
        ax1.legend(fontsize=7, loc='upper right')
        _set_equal_axes(ax1, pred[idx], gt[idx])

        # ── 侧视图（YZ平面）──
        ax2 = fig.add_subplot(2, n_samples, n_samples + col + 1, projection='3d')
        draw_skeleton(ax2, pred[idx], color='royalblue')
        draw_skeleton(ax2, gt[idx],   color='darkorange')
        ax2.view_init(elev=15, azim=20)
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        _set_equal_axes(ax2, pred[idx], gt[idx])

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'vis_ep{epoch}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f'已保存：{save_path}')
    plt.close()


def _set_equal_axes(ax, *point_sets):
    """让三轴等比例，避免骨骼变形"""
    all_pts = np.concatenate(point_sets, axis=0)
    mn, mx = all_pts.min(0), all_pts.max(0)
    mid = (mn + mx) / 2
    r   = max((mx - mn).max() / 2, 0.1) * 1.2
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


def vis_curve():
    """绘制 eval.csv 中的 MPJPE 收敛曲线"""
    import csv
    csv_path = os.path.join(SAVE_DIR, 'eval.csv')
    epochs, mpjpe, pa_mpjpe = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            mpjpe.append(float(row['mpjpe_mm']))
            pa_mpjpe.append(float(row['pa_mpjpe_mm']))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, mpjpe,    label='MPJPE (归一化×1000)', marker='o', ms=3)
    ax.plot(epochs, pa_mpjpe, label='PA-MPJPE',            marker='s', ms=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('误差')
    ax.set_title('SkelKD 训练收敛曲线')
    ax.legend(); ax.grid(alpha=0.3)
    save_path = os.path.join(SAVE_DIR, 'convergence_curve.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f'已保存：{save_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=200, type=int, help='查看哪个 epoch 的结果')
    parser.add_argument('--n',     default=6,   type=int, help='显示样本数')
    parser.add_argument('--curve', action='store_true',   help='显示收敛曲线')
    args = parser.parse_args()

    if args.curve:
        vis_curve()
    else:
        vis_samples(args.epoch, args.n)
        vis_curve()   # 同时生成收敛曲线