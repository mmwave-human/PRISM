"""
diagnose_compressor_latent.py
==============================
诊断 Compressor 潜在空间质量：
  1. 加载已训练的 Compressor checkpoint
  2. 对每个动作提取 latent eps 向量
  3. PCA + t-SNE 可视化 → 判断不同动作是否被区分开

用法：
  python diagnose_compressor_latent.py \
      --ckpt experiments/Compressor_Trainer/mmfi_compressor_202603032012/checkpt_best.pth \
      --data_root data/MMFi/E01 \
      --subject S01 \
      --n_frames 30 \
      --save_dir experiments/Compressor_Trainer/diagnosis

输出：
  diagnosis_pca.png    — PCA 前两主成分散点图
  diagnosis_tsne.png   — t-SNE 散点图
  diagnosis_recon.png  — 各动作随机帧重建质量（CD 误差条形图）
  diagnosis_report.txt — 文字报告：类内/类间距离比
"""

import argparse
import os
import sys
import glob
import re

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 数据 I/O（与 MMFiViPC.py 完全一致）──────────────────────
def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.zeros((1, 3), dtype=np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)

def _normalize(pc):
    c = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s = max(np.max(np.sqrt((pc_c**2).sum(1))), 1e-8)
    return (pc_c / s).astype(np.float32), c.astype(np.float32), float(s)

def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = np.random.choice(N, n, replace=N < n)
    return pc[idx]

# ── CD 计算（简化版，不需要 CUDA 扩展）───────────────────────
def chamfer_distance_simple(p1, p2):
    """(N,3) (M,3) → float，纯 numpy 实现，慢但无依赖"""
    p1 = p1.astype(np.float32)
    p2 = p2.astype(np.float32)
    # p1 → p2
    diff = p1[:, None, :] - p2[None, :, :]   # (N, M, 3)
    dist = (diff**2).sum(-1)                   # (N, M)
    cd1  = dist.min(1).mean()
    cd2  = dist.min(0).mean()
    return float(cd1 + cd2)


# ── 模型加载 ─────────────────────────────────────────────────
def load_compressor(ckpt_path, device):
    """从 checkpoint 内嵌 cfg 加载 Compressor。"""
    from model.Compressor.Network import Compressor
    from tools.io import dict2namespace

    checkpt = torch.load(ckpt_path, map_location='cpu')
    print(f'  checkpoint keys: {list(checkpt.keys())}')

    # 优先从 checkpoint 内嵌的 cfg 获取配置
    if 'cfg' in checkpt:
        raw_cfg = checkpt['cfg']
        cfg = dict2namespace(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
        print(f'  使用 checkpoint 内嵌 cfg, keys: {list(vars(cfg).keys())}')
    else:
        raise RuntimeError('checkpoint 中无 cfg')

    # 找 compressor 模型配置
    comp_cfg = None
    for key in ['model', 'compressor', 'vae', 'network']:
        if hasattr(cfg, key):
            comp_cfg = getattr(cfg, key)
            print(f'  使用 config key: [{key}]')
            break
    if comp_cfg is None:
        raise AttributeError(f'cfg 中无 model/compressor key，现有: {list(vars(cfg).keys())}')

    model = Compressor(comp_cfg).to(device)

    # 加载权重
    state = checkpt.get('state_dict',
            checkpt.get('model_state_dict',
            checkpt.get('compressor_state_dict', None)))
    if state is None:
        raise RuntimeError(f'checkpoint 中无权重，现有: {list(checkpt.keys())}')

    model.load_state_dict(state, strict=True)
    model.init()
    model.eval()
    print(f'  Compressor 加载成功，epoch={checkpt.get("epoch", "?")}')
    return model, cfg


# ── 数据提取 ─────────────────────────────────────────────────
TRAIN_ACTIONS = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']
ACTION_NAMES = {
    'A03': 'Chest_V',   'A12': 'Squat',     'A13': 'RaiseHand_L',
    'A17': 'WaveHand_L','A19': 'PickUp',    'A22': 'Kick_L',
    'A26': 'JumpUp',    'A27': 'Bowing',
}

def collect_frames(data_root, subject, actions, n_frames, lidar_n=256):
    """每个动作随机取 n_frames 帧，返回 lidar 点云列表和对应标签。"""
    all_pcs, all_labels, all_actions = [], [], []

    for act_idx, act in enumerate(actions):
        lidar_dir = os.path.join(data_root, subject, act, 'lidar')
        if not os.path.isdir(lidar_dir):
            print(f'  [跳过] {subject}/{act} 目录不存在')
            continue

        files = glob.glob(os.path.join(lidar_dir, 'frame*_filtered.bin'))
        if not files:
            print(f'  [跳过] {subject}/{act} 无 bin 文件')
            continue

        np.random.shuffle(files)
        collected = 0
        for f in files:
            if collected >= n_frames:
                break
            pc = _load_lidar(f)
            if pc.shape[0] < 32:
                continue
            pc_norm, _, _ = _normalize(pc)
            pc_sub = _subsample(pc_norm, lidar_n)
            all_pcs.append(pc_sub)
            all_labels.append(act_idx)
            all_actions.append(act)
            collected += 1

        print(f'  {act} ({ACTION_NAMES.get(act, act)}): {collected} 帧')

    return np.array(all_pcs, dtype=np.float32), np.array(all_labels), all_actions


# ── 潜在向量提取 ─────────────────────────────────────────────
@torch.no_grad()
def extract_latents(model, pcs_np, device, batch_size=16):
    """pcs_np: (N, P, 3) → latent eps: (N, D)"""
    N = pcs_np.shape[0]
    all_eps = []

    for i in range(0, N, batch_size):
        batch = torch.from_numpy(pcs_np[i:i+batch_size]).float().to(device)
        out   = model(batch)
        # all_eps: (B, D, K) 或 (B, K, D)，flatten 成 (B, -1)
        eps = out['all_eps']          # (B, D, n_layers*z_dim) 或类似
        eps_flat = eps.reshape(eps.shape[0], -1).cpu().numpy()
        all_eps.append(eps_flat)
        if (i // batch_size) % 5 == 0:
            print(f'    提取潜在向量: {min(i+batch_size, N)}/{N}')

    return np.concatenate(all_eps, axis=0)   # (N, D)


# ── 可视化 ───────────────────────────────────────────────────
COLORS = [
    '#E63946','#F4A261','#2A9D8F','#264653',
    '#8338EC','#3A86FF','#FB5607','#06D6A0',
]

def plot_2d(ax, X2d, labels, actions, title, legend=True):
    unique_acts = list(dict.fromkeys(actions))   # 保序去重
    for ai, act in enumerate(unique_acts):
        mask  = np.array(labels) == ai
        color = COLORS[ai % len(COLORS)]
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=color, label=f'{act} {ACTION_NAMES.get(act, "")}',
                   s=40, alpha=0.8, edgecolors='none')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    if legend:
        ax.legend(fontsize=7, loc='best', framealpha=0.7)
    ax.grid(True, alpha=0.3)


def compute_separation(latents, labels):
    """
    计算类内 / 类间距离比（越小越好，说明类间分离度高）。
    ratio < 0.5 → 分离良好；ratio > 0.8 → 几乎无分离
    """
    unique = np.unique(labels)
    centers = {l: latents[labels == l].mean(0) for l in unique}
    global_center = latents.mean(0)

    # 类内距离（intra）
    intra = []
    for l in unique:
        mask = labels == l
        diffs = latents[mask] - centers[l]
        intra.append(np.sqrt((diffs**2).sum(1)).mean())
    intra_mean = np.mean(intra)

    # 类间距离（inter）：各类中心到全局中心
    inter = []
    for l in unique:
        d = np.sqrt(((centers[l] - global_center)**2).sum())
        inter.append(d)
    inter_mean = np.mean(inter)

    ratio = intra_mean / (inter_mean + 1e-8)
    return intra_mean, inter_mean, ratio, {l: centers[l] for l in unique}


def compute_recon_cd(model, pcs_np, labels, device, n_samples=5, batch_size=8):
    """每个动作随机抽 n_samples 帧，计算 Compressor 重建 CD 误差。"""
    unique = np.unique(labels)
    cd_per_action = {}

    for l in unique:
        mask = np.where(labels == l)[0]
        idx  = np.random.choice(mask, min(n_samples, len(mask)), replace=False)
        pcs  = pcs_np[idx]   # (K, P, 3)

        cds = []
        for i in range(0, len(pcs), batch_size):
            batch = torch.from_numpy(pcs[i:i+batch_size]).float().to(device)
            with torch.no_grad():
                out  = model(batch)
                recon = out['set'].cpu().numpy()    # (B, N, 3)
            for j in range(len(batch)):
                cd = chamfer_distance_simple(pcs[i+j], recon[j])
                cds.append(cd)
        cd_per_action[l] = float(np.mean(cds))

    return cd_per_action


# ── 主函数 ───────────────────────────────────────────────────
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. 加载模型
    print('\n[1] 加载 Compressor...')
    model, cfg = load_compressor(args.ckpt, device)

    # 2. 收集数据
    print(f'\n[2] 收集帧数据 ({args.subject}, 每动作 {args.n_frames} 帧)...')
    actions = TRAIN_ACTIONS
    pcs_np, labels, action_seq = collect_frames(
        args.data_root, args.subject, actions, args.n_frames)

    if len(pcs_np) == 0:
        print('没有找到任何数据！请检查 --data_root 和 --subject')
        return

    print(f'  总计 {len(pcs_np)} 帧，动作数 {len(set(action_seq))}')

    # 3. 提取潜在向量
    print('\n[3] 提取潜在向量...')
    latents = extract_latents(model, pcs_np, device)
    print(f'  latent shape: {latents.shape}  (N={latents.shape[0]}, D={latents.shape[1]})')

    # 4. 分离度分析
    print('\n[4] 计算类内/类间距离...')
    intra, inter, ratio, centers = compute_separation(latents, labels)
    print(f'  类内均值距离 (intra): {intra:.4f}')
    print(f'  类间均值距离 (inter): {inter:.4f}')
    print(f'  分离度比值 (intra/inter): {ratio:.4f}')
    if ratio < 0.5:
        verdict = '✅ 分离良好 — Score Network 有足够信息区分动作'
    elif ratio < 0.8:
        verdict = '⚠️  分离一般 — Score Network 学习较困难，建议优先扩充数据'
    else:
        verdict = '❌ 几乎无分离 — Compressor 是瓶颈，需要重新训练'
    print(f'  判断：{verdict}')

    # 5. PCA 可视化
    print('\n[5] PCA 降维...')
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca2d = pca.fit_transform(latents)
    explained = pca.explained_variance_ratio_
    print(f'  PCA 前两成分解释方差: {explained[0]*100:.1f}% + {explained[1]*100:.1f}%')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333366')

    plot_2d(axes[0], pca2d, labels, actions,
            f'PCA (var: {explained[0]*100:.0f}%+{explained[1]*100:.0f}%)')

    # 6. t-SNE 可视化
    print('\n[6] t-SNE 降维 (perplexity=15)...')
    from sklearn.manifold import TSNE
    # 先用 PCA 降到 50 维再 t-SNE，速度更快
    pca50 = PCA(n_components=min(50, latents.shape[1]))
    lat50 = pca50.fit_transform(latents)
    tsne  = TSNE(n_components=2, perplexity=min(15, len(pcs_np)//2),
                 random_state=42, n_iter=1000)
    tsne2d = tsne.fit_transform(lat50)

    plot_2d(axes[1], tsne2d, labels, actions, 't-SNE')

    fig.suptitle(
        f'Compressor 潜在空间诊断  |  {args.subject}  |  '
        f'intra/inter={ratio:.3f}  ({verdict.split("—")[0].strip()})',
        color='white', fontsize=12)
    plt.tight_layout()
    pca_path = os.path.join(args.save_dir, 'diagnosis_pca_tsne.png')
    plt.savefig(pca_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'  已保存: {pca_path}')

    # 7. 重建质量（CD 误差）
    print('\n[7] 计算各动作重建 CD 误差...')
    cd_per_action = compute_recon_cd(model, pcs_np, labels, device,
                                     n_samples=args.n_recon)
    act_names  = [actions[l] for l in sorted(cd_per_action.keys())]
    cd_values  = [cd_per_action[l] for l in sorted(cd_per_action.keys())]

    fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#1a1a2e')
    ax2.set_facecolor('#16213e')
    bars = ax2.bar(range(len(act_names)), cd_values,
                   color=[COLORS[i % len(COLORS)] for i in range(len(act_names))],
                   alpha=0.85)
    ax2.set_xticks(range(len(act_names)))
    ax2.set_xticklabels(
        [f'{a}\n{ACTION_NAMES.get(a,"")}' for a in act_names],
        color='#aaaaaa', fontsize=9)
    ax2.tick_params(colors='#aaaaaa')
    ax2.set_ylabel('Chamfer Distance (重建误差)', color='#aaaaaa')
    ax2.set_title('Compressor 各动作重建质量', color='white', fontsize=12)
    ax2.axhline(y=np.mean(cd_values), color='white', linestyle='--',
                alpha=0.5, label=f'均值={np.mean(cd_values):.5f}')
    ax2.legend(facecolor='#333366', labelcolor='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333366')
    # 标数值
    for bar, v in zip(bars, cd_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{v:.4f}', ha='center', va='bottom', color='white', fontsize=8)
    plt.tight_layout()
    cd_path = os.path.join(args.save_dir, 'diagnosis_recon_cd.png')
    plt.savefig(cd_path, dpi=120, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close()
    print(f'  已保存: {cd_path}')

    # 8. 文字报告
    report_path = os.path.join(args.save_dir, 'diagnosis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('Compressor 潜在空间诊断报告\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Checkpoint : {args.ckpt}\n')
        f.write(f'Subject    : {args.subject}\n')
        f.write(f'总帧数     : {len(pcs_np)}\n')
        f.write(f'Latent dim : {latents.shape[1]}\n\n')

        f.write('── 分离度分析 ──\n')
        f.write(f'  类内均值距离 (intra) : {intra:.4f}\n')
        f.write(f'  类间均值距离 (inter) : {inter:.4f}\n')
        f.write(f'  分离度比值           : {ratio:.4f}\n')
        f.write(f'  判断                 : {verdict}\n\n')

        f.write('── 各动作类内距离 ──\n')
        for l, act in enumerate(actions):
            mask = labels == l
            if mask.sum() == 0:
                continue
            d = np.sqrt(((latents[mask] - centers[l])**2).sum(1)).mean()
            f.write(f'  {act} {ACTION_NAMES.get(act,""):15s}: intra={d:.4f}\n')

        f.write('\n── 各动作重建 CD 误差 ──\n')
        for l in sorted(cd_per_action.keys()):
            act = actions[l]
            f.write(f'  {act} {ACTION_NAMES.get(act,""):15s}: CD={cd_per_action[l]:.5f}\n')
        f.write(f'  均值 CD : {np.mean(cd_values):.5f}\n')

        f.write('\n── PCA 解释方差 ──\n')
        f.write(f'  PC1: {explained[0]*100:.1f}%\n')
        f.write(f'  PC2: {explained[1]*100:.1f}%\n')
        cum = np.cumsum(pca.explained_variance_ratio_)
        f.write(f'  PC1+PC2 累计: {cum[1]*100:.1f}%\n\n')

        f.write('── 解读指南 ──\n')
        f.write('  ratio < 0.5 : 分离良好，Score Network 可正常学习\n')
        f.write('  ratio 0.5~0.8: 分离一般，优先扩充训练动作数量\n')
        f.write('  ratio > 0.8 : Compressor 是瓶颈，需重新训练\n')
        f.write('\n  t-SNE 图中 8 类明显分簇 → 分离良好\n')
        f.write('  t-SNE 图中 8 类混杂 → Compressor 无法区分动作\n')

    print(f'  已保存: {report_path}')
    print(f'\n[诊断完成]')
    print(f'  分离度比值: {ratio:.4f}  →  {verdict}')
    print(f'  查看图表:   {args.save_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compressor 潜在空间诊断')
    parser.add_argument('--ckpt',      required=True,
                        help='Compressor checkpoint 路径')
    parser.add_argument('--data_root', default='data/MMFi/E01',
                        help='MMFi 数据根目录')
    parser.add_argument('--subject',   default='S01',
                        help='受试者，如 S01')
    parser.add_argument('--n_frames',  type=int, default=40,
                        help='每个动作取多少帧')
    parser.add_argument('--n_recon',   type=int, default=10,
                        help='每个动作用于重建 CD 评估的帧数')
    parser.add_argument('--save_dir',
                        default='experiments/Compressor_Trainer/diagnosis')
    parser.add_argument('--gpu',       type=int, default=0)
    args = parser.parse_args()
    main(args)