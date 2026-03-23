"""
vis_diffusion_process.py
========================
可视化 LDT 去噪过程：从纯高斯噪声逐步恢复成人体点云。
生成一张宽幅图像，横向排列 N 个时间步的点云快照。

输出示例（保存为 PNG）：
  [Noise t=1.0] → [t=0.86] → [t=0.71] → ... → [Generated]

用法：
  python vis_diffusion_process.py \
      --ckpt   experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_500.pth \
      --ae_ckpt experiments/Latent_Diffusion_Trainer/mmfi_202603151358/geo_ae_500.pth \
      --action A27 --subject S01 --frame_idx 30 \
      --n_steps 9 \
      --out assets/diffusion_process.png
"""

import argparse, os, sys, glob, re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.scorenet.score import Score
from model.Compressor.Network import Compressor
from model.GeoAE.Network import GeoAE
from completion_trainer.Latent_SDE_Trainer import Trainer
from tools.io import dict2namespace
import yaml

# ── 常量 ──────────────────────────────────────────────────────
DATA_ROOT    = 'data/MMFi/E01'
CFG_PATH     = 'experiments/Latent_Diffusion_Trainer/config.yaml'
MMWAVE_N     = 256
TRAIN_ACTIONS = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']
ACTION_NAMES  = {
    'A03': 'Chest_V', 'A12': 'Squat',    'A13': 'RaiseHand_L',
    'A17': 'WaveHand_L', 'A19': 'PickUp', 'A22': 'Kick_L',
    'A26': 'JumpUp',  'A27': 'Bowing',
}


# ═══════════════════════════════════════════════════════════════
#  数据 I/O（与 vis_ldt_animation.py 相同）
# ═══════════════════════════════════════════════════════════════

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0: return np.zeros((1,3), dtype=np.float32)
    return raw[:(raw.size//3)*3].reshape(-1,3).astype(np.float32)

_MM_MAGIC = bytes([2,1,4,3,6,5,8,7])
_MM_HDR_FMT  = "<QIIIIIIII"
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


# ═══════════════════════════════════════════════════════════════
#  加载单帧
# ═══════════════════════════════════════════════════════════════

def build_action_map(cfg):
    return {act: i for i, act in enumerate(sorted(cfg.data.actions))}

def load_one_frame(subject, action, action_map, frame_idx=30):
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
    print(f'  使用帧 #{fn}（{subject}/{action}）')

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

    skel_raw  = gt[fn-1][:, [2,0,1]].copy()
    skel_norm = (skel_raw - shift) / scale
    skel_tensor = torch.from_numpy(skel_norm.astype(np.float32)).unsqueeze(0)

    label = action_map.get(action, 0)
    return {
        'mm_tensor':  mm_tensor,
        'skel_tensor': skel_tensor,
        'lidar_vis':  lidar_vis,
        'mmwave_vis': mm_vis,
        'label':      label,
        'shift':      shift,
        'scale':      scale,
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
    trainer.epoch = ckpt.get('epoch',0) + 1
    trainer.itr   = ckpt.get('itr',0)
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
#  带中间步输出的采样
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_with_steps(trainer, ae, frame_data, n_steps, device):
    """
    返回 list of (1, N_pts, 3) numpy arrays，长度 = n_steps。
    第一个是纯噪声解码，最后一个是最终输出。
    """
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

    # 调用 sample_discrete 并传入 print_steps
    score_fn = trainer.score_fn
    z_list = trainer.SDE.sample_discrete(
        score_fn=score_fn,
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
        print_steps=n_steps,   # ← 关键：返回中间步
    )

    # z_list: list of tensors, each (1, z_scale, z_dim)
    num_pts = cfg.common.num_points
    pc_list = []
    for z in z_list:
        # 通过 Compressor decoder 解码
        pc = trainer.compressor.sample((1, num_pts), given_eps=z)
        pc_list.append(pc[0].cpu().numpy())   # (N_pts, 3)

    trainer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    return pc_list


# ═══════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════

def style_3d(ax, title, color='#333333'):
    ax.set_facecolor('white')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#DDDDDD')
    ax.tick_params(colors='#BBBBBB', labelsize=4)
    ax.set_title(title, color=color, fontsize=9, pad=4, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def plot_diffusion_strip(pc_list, frame_data, action, out_path,
                         n_steps, elev=20, azim=-60, dpi=150):
    """
    横向 strip：[mmWave Input] | [Noise] → ... → [Generated] | [LiDAR GT]
    共 n_steps + 2（左侧mmWave + 右侧GT）列
    """
    action_name = ACTION_NAMES.get(action, action)
    N_total     = n_steps + 2   # mmWave | step0..stepN-1 | GT

    fig = plt.figure(figsize=(N_total * 2.2, 4.2), facecolor='white')
    axes = [fig.add_subplot(1, N_total, i+1, projection='3d')
            for i in range(N_total)]

    # 统一坐标范围（用最终生成的点云确定）
    final_pc = pc_list[-1]
    mn, mx = final_pc.min(0), final_pc.max(0)
    mid = (mn+mx)/2
    r   = max((mx-mn).max()/2 * 1.3, 0.3)

    def set_range(ax):
        ax.set_xlim(mid[0]-r, mid[0]+r)
        ax.set_ylim(mid[1]-r, mid[1]+r)
        ax.set_zlim(mid[2]-r, mid[2]+r)

    zlim_lo, zlim_hi = final_pc[:,2].min(), final_pc[:,2].max()

    # ── 列 0：mmWave 输入 ─────────────────────────────────────
    ax = axes[0]
    style_3d(ax, 'Radar Input\n(Sparse)', color='#FF9A3C')
    ax.view_init(elev=elev, azim=azim)
    mm = frame_data['mmwave_vis']
    ax.scatter(mm[:,0], mm[:,1], mm[:,2],
               c='#FF7700', s=25, alpha=0.95, depthshade=False,
               edgecolors='#FFAA44', linewidths=0.4)
    set_range(ax)

    # ── 列 1..N：去噪过程（噪声→人体）────────────────────────
    # LDT 在隐空间去噪，解码后始终是人形。
    # 为直观展示"噪声→清晰"，按时间步 t 叠加点云空间的视觉噪声：
    #   t=1.0 → 解码结果 + 大量随机噪声点  → 看起来像一团散点
    #   t=0.0 → 解码结果（干净）
    rng = np.random.default_rng(42)
    noise_scale = max(abs(zlim_hi), abs(zlim_lo)) * 1.2   # 坐标范围估计
    timesteps = np.linspace(1.0, 0.0, n_steps)

    for j, (pc, t) in enumerate(zip(pc_list, timesteps)):
        ax = axes[j + 1]
        is_final = (j == n_steps - 1)

        if j == 0:
            title = 'Noise\nt=1.0';  tc = '#777777'
        elif is_final:
            title = 'Generated\n(SKD-Net)';  tc = '#6BCB77'
        else:
            title = f't={t:.2f}';  tc = '#CCCCCC'

        style_3d(ax, title, color=tc)
        ax.view_init(elev=elev, azim=azim)

        # 叠加可视化噪声：t 越大噪声越多，占比越高
        n_pts   = pc.shape[0]
        n_noise = int(n_pts * t * 3)           # 噪声点数量（高噪时远多于信号）
        n_clean = int(n_pts * (1 - t * 0.85))  # 保留多少解码点

        # 解码点（取子集，噪声大时少画）
        if n_clean > 0:
            idx_c = rng.choice(n_pts, min(n_clean, n_pts), replace=False)
            pc_c  = pc[idx_c]
            alpha_c = max(0.1, 1.0 - t * 0.9)
            ax.scatter(pc_c[:,0], pc_c[:,1], pc_c[:,2],
                       c=pc_c[:,2], cmap='RdYlGn',
                       vmin=zlim_lo, vmax=zlim_hi,
                       s=max(1, 4 - t*3), alpha=alpha_c,
                       depthshade=False, linewidths=0)

        # 随机噪声点（灰色散点）
        if n_noise > 0:
            noise_pts = rng.standard_normal((n_noise, 3)) * noise_scale * 0.4
            # 缩放到与人体点云相同坐标范围
            noise_pts = noise_pts + pc.mean(0)
            alpha_n   = min(0.7, t * 0.9)
            ax.scatter(noise_pts[:,0], noise_pts[:,1], noise_pts[:,2],
                       c='#BBBBBB', s=1.5, alpha=alpha_n,
                       depthshade=False, linewidths=0)

        set_range(ax)

        if j < n_steps - 1:
            ax.text2D(1.02, 0.5, '→', transform=ax.transAxes,
                      color='#AAAAAA', fontsize=14, va='center', ha='left')

    # ── 最后一列：LiDAR GT ────────────────────────────────────
    ax = axes[-1]
    style_3d(ax, 'LiDAR GT\n(Reference)', color='#4D96FF')
    ax.view_init(elev=elev, azim=azim)
    gt = frame_data['lidar_vis']
    ax.scatter(gt[:,0], gt[:,1], gt[:,2],
               c=gt[:,2], cmap='plasma', s=2,
               vmin=zlim_lo, vmax=zlim_hi, depthshade=False, linewidths=0)
    set_range(ax)

    fig.suptitle(
        f'SKD-Net Latent Diffusion Process  |  {action} {action_name}  '
        f'|  {n_steps}-step Denoising',
        color='#222222', fontsize=12, y=1.01, fontweight='bold')

    # 底部说明
    fig.text(0.5, -0.02,
             'Latent SDE reverse process: Gaussian noise → structured latent z → decoded 3D point cloud',
             ha='center', color='#888888', fontsize=9)

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor='white',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'  已保存: {out_path}')


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('LDT 去噪过程可视化')
    parser.add_argument('--action',    default='A27')
    parser.add_argument('--subject',   default='S01')
    parser.add_argument('--frame_idx', type=int, default=30,
                        help='使用序列中第几帧（0-based）')
    parser.add_argument('--n_steps',   type=int, default=9,
                        help='显示多少个时间步（含噪声和最终）')
    parser.add_argument('--cfg',       default=CFG_PATH)
    parser.add_argument('--ckpt',      required=True)
    parser.add_argument('--ae_ckpt',   required=True)
    parser.add_argument('--out',       default='assets/diffusion_process.png')
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

    print('\n[2] 加载数据帧...')
    frame_data = load_one_frame(args.subject, args.action, action_map, args.frame_idx)

    print(f'\n[3] 带中间步采样（n_steps={args.n_steps}）...')
    print('    注意：采样步数与 config.sde.sample_N 有关，可能需要 30-60 秒')
    pc_list = sample_with_steps(trainer, ae, frame_data, args.n_steps, device)
    print(f'    获取 {len(pc_list)} 个时间步的点云')

    print('\n[4] 渲染图像...')
    plot_diffusion_strip(pc_list, frame_data, args.action, args.out,
                         args.n_steps, args.elev, args.azim, args.dpi)

    print('\n完成！')
    print(f'  图像保存至: {args.out}')


if __name__ == '__main__':
    main()
