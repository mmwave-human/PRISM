"""
vis_web_demo.py
===============
生成网页展示用的 Demo 合成视频：
  [Video RGB] | [Video Depth] | [mmWave Input] | [Generated PC] | [SkelKD Skeleton]

上方两格从 assets/MMFi_data_demo.mp4 抽帧（RGB + 灰度模拟 Depth），
下方三格为真实雷达数据 + LDT/SkelKD 模型推理结果。

用法：
  python vis_web_demo.py \
      --ckpt   experiments/Latent_Diffusion_Trainer/mmfi_202603151358/checkpt_500.pth \
      --ae_ckpt experiments/Latent_Diffusion_Trainer/mmfi_202603151358/geo_ae_500.pth \
      --skelkd_ckpt experiments/SkelKD/mmfi_skelkd_202603091941/checkpt_best.pth \
      --action A27 --subject S01 --start 0 --end 60 \
      --out assets/demo_web.mp4
"""

import argparse, os, sys, glob, re, time, subprocess
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter
import io
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.scorenet.score import Score
from model.Compressor.Network import Compressor
from model.GeoAE.Network import GeoAE
from model.SkelKD.Network import SkelKD
from completion_trainer.Latent_SDE_Trainer import Trainer
from tools.io import dict2namespace
import yaml

# ── 常量 ──────────────────────────────────────────────────────
DATA_ROOT     = 'data/MMFi/E01'
LDT_CFG_PATH  = 'experiments/Latent_Diffusion_Trainer/config.yaml'
SKEL_CFG_PATH = 'experiments/SkelKD/config.yaml'
DEMO_VIDEO    = 'assets/MMFi_data_demo.mp4'
MMWAVE_N      = 256
LIDAR_N       = 256
SKEL_MM_N     = 100

TRAIN_ACTIONS = ['A03', 'A12', 'A13', 'A17', 'A19', 'A22', 'A26', 'A27']
ACTION_NAMES  = {
    'A03': 'Chest_V',    'A12': 'Squat',       'A13': 'RaiseHand_L',
    'A17': 'WaveHand_L', 'A19': 'PickUp',       'A22': 'Kick_L',
    'A26': 'JumpUp',     'A27': 'Bowing',
}
BONES_LDT = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),(8,14),(14,15),(15,16),
]
BONE_COLORS_LDT = {
    (0,1):'#FF9A3C',(1,2):'#FF9A3C',(2,3):'#FF9A3C',
    (0,4):'#C77DFF',(4,5):'#C77DFF',(5,6):'#C77DFF',
    (0,7):'#FFD93D',(7,8):'#FFD93D',(8,9):'#FFD93D',(9,10):'#FF6B6B',
    (8,11):'#6BCB77',(11,12):'#6BCB77',(12,13):'#6BCB77',
    (8,14):'#4D96FF',(14,15):'#4D96FF',(15,16):'#4D96FF',
}


# ═══════════════════════════════════════════════════════════════
#  数据 I/O（复用 vis_ldt_animation.py）
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
        if idx + _MM_HDR_SIZE > len(raw): break
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
#  序列加载
# ═══════════════════════════════════════════════════════════════

def build_action_map(cfg):
    actions = list(cfg.data.actions)
    return {act: i for i, act in enumerate(sorted(actions))}

def load_sequence(subject, action, action_map, start=0, end=60):
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
    elif gt_raw.ndim==4 and s[0]==1:  gt = gt_raw[0].astype(np.float32)
    else: raise ValueError(f'无法解析 GT: {gt_raw.shape}')

    l_files = {_frame_num(f): f for f in glob.glob(os.path.join(lidar_dir, 'frame*_filtered.bin'))}
    m_files = {_frame_num(f): f for f in glob.glob(os.path.join(mmwave_dir,'frame*_filtered.bin'))}
    l_files.pop(-1, None); m_files.pop(-1, None)
    common = sorted(set(l_files) & set(m_files))
    common = [fn for fn in common if fn-1 < len(gt)][start:end]
    if not common: raise RuntimeError(f'无有效帧：{subject}/{action}')
    print(f'  加载 {subject}/{action}，{len(common)} 帧')

    label  = action_map.get(action, 0)
    frames = []
    for fn in common:
        lidar = _load_lidar(l_files[fn])
        lidar_norm, shift, scale = _normalize(lidar)
        lidar_vis = _subsample(lidar_norm, LIDAR_N)

        mm_xyz = _load_mmwave(m_files[fn])
        valid  = np.isfinite(mm_xyz).all(1) & (np.abs(mm_xyz)<20.).all(1)
        mm_xyz = mm_xyz[valid]
        if mm_xyz.shape[0] == 0: mm_xyz = np.zeros((1,3), dtype=np.float32)
        mm_norm, _, _ = _normalize(mm_xyz)
        mm_vis    = _subsample(mm_norm, MMWAVE_N)
        mm_tensor = torch.from_numpy(mm_vis).float().unsqueeze(0)

        # SkelKD 用的 mmWave（无 normalize，subsample 到 SKEL_MM_N）
        mm_skel = _subsample(mm_norm, SKEL_MM_N)
        mm_skel_t = torch.from_numpy(mm_skel).float().unsqueeze(0)

        skel_raw    = gt[fn-1][:, [2,0,1]].copy()
        skel_norm   = (skel_raw - shift) / scale
        skel_tensor = torch.from_numpy(skel_norm.astype(np.float32)).unsqueeze(0)
        skel_vis    = skel_norm.copy()
        skel_vis[:,2] *= -1

        frames.append({
            'lidar_vis':   lidar_vis,
            'mmwave_vis':  mm_vis,
            'mm_tensor':   mm_tensor,
            'mm_skel_t':   mm_skel_t,
            'gt_skel_vis': skel_vis,
            'skel_tensor': skel_tensor,
            'label':       label,
            'frame_num':   fn,
        })
    return frames


# ═══════════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════════

def load_ldt(cfg_path, ckpt_path, ae_ckpt_path, device):
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
        feat_dim=getattr(ae_cfg, 'feat_dim', 256) if ae_cfg else 256,
        num_coarse=getattr(ae_cfg, 'num_coarse', 128) if ae_cfg else 128,
    ).to(device)
    if ae_ckpt_path and os.path.exists(ae_ckpt_path):
        ae.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
        print(f'  AE loaded from {ae_ckpt_path}')
    ae.eval()
    return trainer, ae, cfg

def load_skelkd(cfg_path, ckpt_path, device):
    with open(cfg_path) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader))
    model = SkelKD(cfg.model).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'  SkelKD loaded epoch={ckpt.get("epoch","?")}')
    return model


# ═══════════════════════════════════════════════════════════════
#  推理
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(trainer, ae, skelkd, frames, device):
    gen_pcs, pred_skels = [], []
    for i, fr in enumerate(frames):
        mm    = fr['mm_tensor'].to(device)
        skel  = fr['skel_tensor'].to(device)
        label = torch.tensor([fr['label']], dtype=torch.long).to(device)
        coarse, _ = ae(mm)
        smp, _ = trainer.sample(1, label=label,
                                condition={'skeleton': skel, 'coarse': coarse})
        gen_pcs.append(smp[0].cpu().numpy())

        mm_sk = fr['mm_skel_t'].to(device)
        pred  = skelkd(mm_sk).squeeze(0).cpu().numpy()
        pred_skels.append(pred)

        if (i+1) % 10 == 0:
            print(f'  推理 {i+1}/{len(frames)}')
    return gen_pcs, pred_skels


# ═══════════════════════════════════════════════════════════════
#  视频帧提取
# ═══════════════════════════════════════════════════════════════

def extract_video_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total-1, n_frames, dtype=int)
    frames_rgb = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frames_rgb.append(np.zeros((360,640,3), dtype=np.uint8))
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
    cap.release()
    print(f'  提取 {len(frames_rgb)} 帧视频（共 {total} 帧原视频）')
    return frames_rgb


# ═══════════════════════════════════════════════════════════════
#  绘图工具
# ═══════════════════════════════════════════════════════════════

def style_3d(ax, title):
    ax.set_facecolor('white')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#DDDDDD')
    ax.tick_params(colors='#BBBBBB', labelsize=4)
    ax.set_title(title, color='#333333', fontsize=9, pad=3, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def _global_range(arr_list, pad=1.2):
    pts = np.concatenate([a.reshape(-1,3) for a in arr_list])
    mn, mx = pts.min(0), pts.max(0)
    mid = (mn+mx)/2
    r   = max((mx-mn).max()/2*pad, 0.3)
    return mid, r

def draw_skel(ax, joints, bones, bone_colors):
    for (a,b) in bones:
        c = bone_colors.get((a,b), bone_colors.get((b,a), '#888888'))
        ax.plot([joints[a,0],joints[b,0]],[joints[a,1],joints[b,1]],[joints[a,2],joints[b,2]],
                color=c, lw=1.8)
    ax.scatter(joints[:,0], joints[:,1], joints[:,2],
               c='#333333', s=10, zorder=5, edgecolors='#777777', linewidths=0.4)


# ═══════════════════════════════════════════════════════════════
#  合成帧渲染（matplotlib → numpy image）
# ═══════════════════════════════════════════════════════════════

def render_frame(video_rgb, fr, gen_pc, pred_skel,
                 mid_mm, r_mm, mid_gen, r_gen, mid_sk, r_sk,
                 zlim_lo, zlim_hi, action, action_name, frame_idx, total_frames,
                 elev=15, azim=-65, dpi=150):

    fig = plt.figure(figsize=(16, 5.5), facecolor='white')
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                            left=0.01, right=0.99, top=0.88, bottom=0.01,
                            wspace=0.08, hspace=0.15,
                            width_ratios=[1.4, 1.4, 1, 1, 1],
                            height_ratios=[1, 1])

    # ── 上方：视频帧 (RGB) ─────────────────────────────────────
    ax_rgb  = fig.add_subplot(gs[0, :2])
    ax_gray = fig.add_subplot(gs[1, :2])

    h, w = video_rgb.shape[:2]
    half = w // 2
    ax_rgb.imshow(video_rgb[:, :half, :])
    ax_rgb.set_title('RGB Video', color='#333333', fontsize=8, pad=2)
    ax_rgb.axis('off')

    gray = cv2.cvtColor(video_rgb[:, half:, :], cv2.COLOR_RGB2GRAY)
    ax_gray.imshow(gray, cmap='magma')
    ax_gray.set_title('Depth (simulated)', color='#333333', fontsize=8, pad=2)
    ax_gray.axis('off')

    # ── 右侧：3D 点云 ──────────────────────────────────────────
    ax_mm  = fig.add_subplot(gs[:, 2], projection='3d')
    ax_gen = fig.add_subplot(gs[:, 3], projection='3d')
    ax_sk  = fig.add_subplot(gs[:, 4], projection='3d')

    for ax, title in [(ax_mm, 'Radar Input'),
                      (ax_gen, 'Generated PC\n(SKD-Net)'),
                      (ax_sk,  'Skeleton\n(SkelKD)')]:
        style_3d(ax, title)
        ax.view_init(elev=elev, azim=azim)

    # mmWave
    mm = fr['mmwave_vis']
    ax_mm.scatter(mm[:,0], mm[:,1], mm[:,2],
                  c='#FF7700', s=18, alpha=0.9, depthshade=False,
                  edgecolors='#FFAA44', linewidths=0.3)
    ax_mm.set_xlim(mid_mm[0]-r_mm, mid_mm[0]+r_mm)
    ax_mm.set_ylim(mid_mm[1]-r_mm, mid_mm[1]+r_mm)
    ax_mm.set_zlim(mid_mm[2]-r_mm, mid_mm[2]+r_mm)

    # Generated PC
    ax_gen.scatter(gen_pc[:,0], gen_pc[:,1], gen_pc[:,2],
                   c=gen_pc[:,2], cmap='RdYlGn', s=3,
                   vmin=zlim_lo, vmax=zlim_hi, depthshade=False, linewidths=0)
    ax_gen.set_xlim(mid_gen[0]-r_gen, mid_gen[0]+r_gen)
    ax_gen.set_ylim(mid_gen[1]-r_gen, mid_gen[1]+r_gen)
    ax_gen.set_zlim(mid_gen[2]-r_gen, mid_gen[2]+r_gen)

    # SkelKD skeleton
    sk = pred_skel.copy(); sk[:,2] *= -1
    draw_skel(ax_sk, sk, BONES_LDT, BONE_COLORS_LDT)
    ax_sk.set_xlim(mid_sk[0]-r_sk, mid_sk[0]+r_sk)
    ax_sk.set_ylim(mid_sk[1]-r_sk, mid_sk[1]+r_sk)
    ax_sk.set_zlim(mid_sk[2]-r_sk, mid_sk[2]+r_sk)

    fig.suptitle(
        f'SKD-Net Demo  |  Action: {action} {action_name}  |  Frame {frame_idx+1}/{total_frames}',
        color='#222222', fontsize=11, y=0.96, fontweight='bold')

    # 渲染为 numpy 数组
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf))
    return img[:, :, :3]   # drop alpha


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('SKD-Net Web Demo Video')
    parser.add_argument('--action',      default='A27')
    parser.add_argument('--subject',     default='S01')
    parser.add_argument('--start',       type=int, default=0)
    parser.add_argument('--end',         type=int, default=60)
    parser.add_argument('--cfg',         default=LDT_CFG_PATH)
    parser.add_argument('--ckpt',        required=True)
    parser.add_argument('--ae_ckpt',     required=True)
    parser.add_argument('--skelkd_ckpt', required=True)
    parser.add_argument('--demo_video',  default=DEMO_VIDEO)
    parser.add_argument('--out',         default='assets/demo_web.mp4')
    parser.add_argument('--fps',         type=int, default=8)
    parser.add_argument('--elev',        type=float, default=15)
    parser.add_argument('--azim',        type=float, default=-65)
    parser.add_argument('--dpi',         type=int,   default=100)
    parser.add_argument('--gpu',         type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 加载模型
    print('\n[1] 加载模型...')
    trainer, ae, cfg = load_ldt(args.cfg, args.ckpt, args.ae_ckpt, device)
    skelkd = load_skelkd(SKEL_CFG_PATH, args.skelkd_ckpt, device)
    action_map = build_action_map(cfg)

    # 加载数据
    print('\n[2] 加载雷达数据...')
    frames = load_sequence(args.subject, args.action, action_map,
                           args.start, args.end)
    N = len(frames)

    # 推理
    print('\n[3] 模型推理...')
    gen_pcs, pred_skels = run_inference(trainer, ae, skelkd, frames, device)

    # 提取视频帧
    print('\n[4] 提取 Demo 视频帧...')
    video_frames = extract_video_frames(args.demo_video, N)

    # 预计算坐标范围（固定轴）
    mid_mm,  r_mm  = _global_range([fr['mmwave_vis'] for fr in frames])
    mid_gen, r_gen = _global_range(gen_pcs + [fr['lidar_vis'] for fr in frames])
    mid_sk,  r_sk  = _global_range([ps.copy() for ps in pred_skels])
    zlim_lo = min(g[:,2].min() for g in gen_pcs)
    zlim_hi = max(g[:,2].max() for g in gen_pcs)

    action_name = ACTION_NAMES.get(args.action, args.action)

    # 渲染第一帧以获取输出尺寸
    print('\n[5] 渲染合成帧...')
    sample_frame = render_frame(video_frames[0], frames[0], gen_pcs[0], pred_skels[0],
                                mid_mm, r_mm, mid_gen, r_gen, mid_sk, r_sk,
                                zlim_lo, zlim_hi, args.action, action_name,
                                0, N, args.elev, args.azim, args.dpi)
    H, W = sample_frame.shape[:2]
    # H.264 要求宽高为偶数，向下裁剪
    W = W - (W % 2)
    H = H - (H % 2)
    sample_frame = sample_frame[:H, :W]
    print(f'  输出分辨率: {W}×{H}')

    # 用 ffmpeg pipe 写入 H.264 MP4（浏览器/VS Code 兼容）
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'rgb24',
        '-r', str(args.fps),
        '-i', 'pipe:0',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '20',
        '-preset', 'fast',
        args.out,
    ]
    ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ffmpeg_proc.stdin.write(sample_frame.tobytes())
    for i in range(1, N):
        img = render_frame(video_frames[i], frames[i], gen_pcs[i], pred_skels[i],
                           mid_mm, r_mm, mid_gen, r_gen, mid_sk, r_sk,
                           zlim_lo, zlim_hi, args.action, action_name,
                           i, N, args.elev, args.azim, args.dpi)
        ffmpeg_proc.stdin.write(img[:H, :W].tobytes())
        if (i+1) % 10 == 0:
            print(f'  渲染进度: {i+1}/{N}')

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f'\n完成！已保存: {args.out}')
    print(f'  {N} 帧 @ {args.fps}fps = {N/args.fps:.1f}s, 分辨率 {W}×{H} (H.264)')


if __name__ == '__main__':
    main()
