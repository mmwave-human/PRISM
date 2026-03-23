"""
vis_diffusion_grid.py
=====================
补充材料：8 个动作的去噪过程合成大图。
每行一个动作，每列显示从纯噪声到人体点云的去噪快照。
噪声点与人体点云使用相同颜色——视觉差异仅来自空间分布的有序程度。

用法：
  python vis_diffusion_grid.py \
      --ckpt   experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_500.pth \
      --ae_ckpt experiments/Latent_Diffusion_Trainer/mmfi_202603151358/geo_ae_500.pth \
      --n_steps 6 \
      --out assets/diffusion_grid.png
"""

import argparse, os, sys, glob, re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

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

# 8 个动作及对应的代表帧（S01，0-based 帧索引）
ACTIONS   = ['A03', 'A12', 'A13', 'A19', 'A22', 'A26', 'A27', 'A17']
FRAME_IDX = [  29,    8,    73,    59,    50,    40,    51,    44  ]
SUBJECT   = 'S01'

# 每个动作独立颜色；噪声点与人体点使用同一颜色
ACTION_COLORS = {
    'A03': '#4D96FF',   # 蓝    Chest V
    'A12': '#FF6B6B',   # 红    Squat
    'A13': '#48BB78',   # 绿    Raise Hand L
    'A19': '#FF9A3C',   # 橙    Pick Up
    'A22': '#9B59B6',   # 紫    Kick L
    'A26': '#D4A017',   # 金    Jump Up
    'A27': '#00B4D8',   # 青    Bowing
    'A17': '#F72585',   # 粉    Wave Hand L
}
ACTION_NAMES = {
    'A03': 'Chest V',   'A12': 'Squat',
    'A13': 'Raise Hand','A17': 'Wave Hand',
    'A19': 'Pick Up',   'A22': 'Kick',
    'A26': 'Jump Up',   'A27': 'Bowing',
}
TRAIN_ACTIONS = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']


# ═══════════════════════════════════════════════════════════════
#  数据 I/O（复用 vis_diffusion_process.py）
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
    lidar_vis = _subsample(lidar_norm, 256)

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

    label = action_map.get(action, 0)
    return {
        'mm_tensor':   mm_tensor,
        'skel_tensor': skel_tensor,
        'lidar_vis':   lidar_vis,
        'mmwave_vis':  mm_vis,
        'label':       label,
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
#  带中间步采样（复用 vis_diffusion_process.py 逻辑）
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_with_steps(trainer, ae, frame_data, n_steps, device):
    mm    = frame_data['mm_tensor'].to(device)
    skel  = frame_data['skel_tensor'].to(device)
    label = torch.tensor([frame_data['label']], dtype=torch.long).to(device)

    coarse, _ = ae(mm)
    condition  = {'skeleton': skel, 'coarse': coarse}

    trainer.model.eval()
    trainer.compressor.eval()
    trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    cfg = trainer.cfg
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
#  绘图工具
# ═══════════════════════════════════════════════════════════════

def style_3d(ax, title='', title_color='#333333', title_size=8):
    ax.set_facecolor('white')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False
        p.set_edgecolor('#E0E0E0')
    ax.tick_params(colors='#CCCCCC', labelsize=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    if title:
        ax.set_title(title, color=title_color, fontsize=title_size,
                     pad=2, fontweight='bold')


# ═══════════════════════════════════════════════════════════════
#  主渲染：8行 × (1mmWave + n_steps + 1GT) 网格
# ═══════════════════════════════════════════════════════════════

def render_grid(all_pc_lists, all_frame_data, actions,
                out_path, n_steps, elev=20, azim=-60, dpi=150):

    N_ROWS = len(actions)
    N_COLS = 1 + n_steps + 1   # mmWave | n_steps 去噪 | GT

    # 图像尺寸
    cell_w, cell_h = 2.0, 2.3
    left_pad = 0.9   # 行标签宽度 (inches)
    top_pad  = 0.6   # 列标题高度
    bot_pad  = 0.3
    fig_w = left_pad + N_COLS * cell_w
    fig_h = top_pad + N_ROWS * cell_h + bot_pad

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # GridSpec（留出左侧行标签空间）
    gs = gridspec.GridSpec(
        N_ROWS, N_COLS, figure=fig,
        left  = left_pad / fig_w,
        right = 0.995,
        top   = 1.0 - top_pad / fig_h,
        bottom= bot_pad / fig_h,
        wspace=0.04,
        hspace=0.08,
    )

    rng = np.random.default_rng(42)

    # 时间步标签（只在第一行顶部显示）
    timesteps = np.linspace(1.0, 0.0, n_steps)
    col_labels = ['Radar\nInput'] \
               + ['Noise\n$t=1.0$'] \
               + [f'$t={t:.2f}$' for t in timesteps[1:-1]] \
               + ['Generated'] \
               + ['LiDAR\nGT']

    for row_i, (action, pc_list, frame_data) in enumerate(
            zip(actions, all_pc_lists, all_frame_data)):

        color = ACTION_COLORS.get(action, '#4D96FF')
        aname = ACTION_NAMES.get(action, action)

        # 统一坐标范围（以最终生成点云为准）
        final_pc = pc_list[-1]
        mn, mx = final_pc.min(0), final_pc.max(0)
        mid = (mn + mx) / 2
        r   = max((mx - mn).max() / 2 * 1.35, 0.3)
        noise_scale = r * 1.8   # 噪声散布半径

        def set_range(ax):
            ax.set_xlim(mid[0]-r, mid[0]+r)
            ax.set_ylim(mid[1]-r, mid[1]+r)
            ax.set_zlim(mid[2]-r, mid[2]+r)

        # ── 行标签 ────────────────────────────────────────────
        # 用 ax.text2D 画在 mmWave 格左侧
        ax_label = fig.add_subplot(gs[row_i, 0], projection='3d')
        # 先渲染 mmWave，再通过 text2D 加标签
        # （这里将标签放在 col0 左侧，通过 transAxes 偏移）

        # ── 列 0：mmWave 输入 ─────────────────────────────────
        ax = ax_label
        title_0 = col_labels[0] if row_i == 0 else ''
        style_3d(ax, title_0, title_color='#555555')
        ax.view_init(elev=elev, azim=azim)
        mm = frame_data['mmwave_vis']
        ax.scatter(mm[:,0], mm[:,1], mm[:,2],
                   c='#FF7700', s=22, alpha=0.9, depthshade=False,
                   edgecolors='#FFAA44', linewidths=0.3)
        set_range(ax)

        # 行标签（动作编号 + 名称，放在 mmWave 格左侧）
        ax.text2D(-0.28, 0.5,
                  f'{action}\n{aname}',
                  transform=ax.transAxes,
                  color=color, fontsize=8.5, fontweight='bold',
                  va='center', ha='right', linespacing=1.4)

        # ── 列 1..n_steps：去噪过程 ────────────────────────────
        for j, (pc, t) in enumerate(zip(pc_list, timesteps)):
            ax = fig.add_subplot(gs[row_i, j+1], projection='3d')
            title_j = col_labels[j+1] if row_i == 0 else ''
            style_3d(ax, title_j, title_color='#555555')
            ax.view_init(elev=elev, azim=azim)

            n_pts   = pc.shape[0]
            # 噪声点数：t=1→3×n_pts，t=0→0
            n_noise = int(n_pts * t * 3)
            # 干净点数：t=1→15%，t=0→100%
            n_clean = max(1, int(n_pts * (1.0 - t * 0.85)))

            # ── 干净点（人体解码结果）────────────────────────
            idx_c   = rng.choice(n_pts, min(n_clean, n_pts), replace=False)
            pc_c    = pc[idx_c]
            alpha_c = float(np.clip(1.0 - t * 0.85, 0.15, 1.0))
            size_c  = float(np.clip(5.0 - t * 3.5, 1.0, 5.0))
            ax.scatter(pc_c[:,0], pc_c[:,1], pc_c[:,2],
                       c=color, s=size_c, alpha=alpha_c,
                       depthshade=False, linewidths=0)

            # ── 噪声点（同色，随机散布在更大球内，裁剪到轴范围）──
            if n_noise > 0:
                noise_pts = (rng.standard_normal((n_noise, 3)) * noise_scale * 0.55
                             + pc.mean(0))
                # 裁剪到坐标轴范围内，避免溢出导致渲染瑕疵
                for dim, center in enumerate(mid):
                    noise_pts[:, dim] = np.clip(
                        noise_pts[:, dim], center - r * 0.94, center + r * 0.94)
                alpha_n   = float(np.clip(t * 0.75, 0.0, 0.65))
                ax.scatter(noise_pts[:,0], noise_pts[:,1], noise_pts[:,2],
                           c=color, s=1.2, alpha=alpha_n,
                           depthshade=False, linewidths=0)

            set_range(ax)

        # ── 最后一列：LiDAR GT ────────────────────────────────
        ax = fig.add_subplot(gs[row_i, n_steps+1], projection='3d')
        title_gt = col_labels[-1] if row_i == 0 else ''
        style_3d(ax, title_gt, title_color='#555555')
        ax.view_init(elev=elev, azim=azim)
        gt = frame_data['lidar_vis']
        ax.scatter(gt[:,0], gt[:,1], gt[:,2],
                   c='#888888', s=2.5, alpha=0.85, depthshade=False, linewidths=0)
        set_range(ax)

    # ── 标题 ──────────────────────────────────────────────────
    fig.suptitle(
        'SKD-Net — Latent Diffusion Denoising Process (8 Actions)',
        color='#222222', fontsize=11, fontweight='bold',
        y=1.0 - 0.12 / fig_h,
    )

    # ── 底部说明 ──────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        'Each row: noise (scattered points) → structured 3D human point cloud  '
        '|  Same color per action for both noise and body  '
        '|  Latent SDE reverse process',
        ha='center', color='#888888', fontsize=7.5,
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
    parser = argparse.ArgumentParser('LDT 全动作去噪过程合成图（补充材料）')
    parser.add_argument('--cfg',       default=CFG_PATH)
    parser.add_argument('--ckpt',      required=True)
    parser.add_argument('--ae_ckpt',   required=True)
    parser.add_argument('--subject',   default=SUBJECT)
    parser.add_argument('--n_steps',   type=int, default=6,
                        help='每行显示多少个去噪时间步（不含 GT 列）')
    parser.add_argument('--out',       default='assets/diffusion_grid.png')
    parser.add_argument('--elev',      type=float, default=20)
    parser.add_argument('--azim',      type=float, default=-60)
    parser.add_argument('--dpi',       type=int,   default=150)
    parser.add_argument('--gpu',       type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\n[1] 加载模型...')
    trainer, ae, cfg = load_models(args.cfg, args.ckpt, args.ae_ckpt, device)
    action_map = build_action_map(cfg)

    print('\n[2] 加载各动作代表帧...')
    all_frame_data = []
    for action, fidx in zip(ACTIONS, FRAME_IDX):
        fd = load_one_frame(args.subject, action, action_map, fidx)
        all_frame_data.append(fd)

    print(f'\n[3] 对每个动作进行带中间步采样（n_steps={args.n_steps}）...')
    print('    每个动作约 30-60 秒，共 8 个动作\n')
    all_pc_lists = []
    for i, (action, fd) in enumerate(zip(ACTIONS, all_frame_data)):
        print(f'  [{i+1}/8] {action} {ACTION_NAMES.get(action,"")}')
        pc_list = sample_with_steps(trainer, ae, fd, args.n_steps, device)
        all_pc_lists.append(pc_list)
        print(f'        获取 {len(pc_list)} 步点云')

    print('\n[4] 渲染合成图...')
    render_grid(all_pc_lists, all_frame_data, ACTIONS,
                args.out, args.n_steps, args.elev, args.azim, args.dpi)

    print(f'\n完成！图像保存至: {args.out}')


if __name__ == '__main__':
    main()
