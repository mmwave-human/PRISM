"""
datasets/MISTViPC.py
====================
MIST 数据集加载器 — PRISM 的 MIST 数据接口。

与 MMFiViPC.py 的主要差异：
  ① 目录层级：OL/{subj}/  （无动作子目录；动作标签来自 action_labels.npy）
  ② LiDAR 格式：frame*.pcd（兼容 .bin）
  ③ 骨骼文件：skeleton.npy（替代 ground_truth.npy）
  ④ 主体命名：P01–P08；动作命名：R01–R14 / D01–D16
  ⑤ 遮挡等级：OL0 / OL1 / OL2（可组合加载）

目录结构：
    MIST/
    ├── calibration/extrinsics.json
    ├── OL0/
    │   ├── P01/
    │   │   ├── mmwave/         frame{n:04d}.bin
    │   │   ├── lidar/          frame{n:04d}.{bin,pcd}
    │   │   ├── skeleton.npy    (N, 17, 3)  H36M 关节，相机世界坐标
    │   │   ├── action_labels.npy  (N,) int  每帧动作索引（见 ACTION_INDEX）
    │   │   ├── rgb/
    │   │   └── depth/
    │   ├── P02/ ... P08/
    ├── OL1/
    └── OL2/

返回四元组 (skeleton, lidar, mmwave, label)，与 MMFiViPC 接口完全一致。

坐标系对齐约定（与 MMFiViPC.py 相同）：
    skeleton.npy 原始坐标系（相机世界）：(lateral, height, depth)
    目标坐标系（LiDAR/mmWave 约定）：  (depth, lateral, height)
    变换：skel[:, [2, 0, 1]]，然后用 LiDAR 的 shift/scale 归一化。
"""

import os, re, glob, random, json, struct
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils import data as tdata


# ──────────────────────────────────────────────────────────
#  动作集合 & 索引映射
# ──────────────────────────────────────────────────────────

REHAB_ACTIONS = [f'R{i:02d}' for i in range(1, 15)]   # R01–R14  (14 个)
DAILY_ACTIONS = [f'D{i:02d}' for i in range(1, 17)]   # D01–D16  (16 个)
ALL_ACTIONS   = REHAB_ACTIONS + DAILY_ACTIONS           # 30 个，R01=0 … D16=29

# action_labels.npy 中整数值与动作名称的对应关系
ACTION_INDEX = {act: i for i, act in enumerate(ALL_ACTIONS)}  # {'R01':0, ..., 'D16':29}
INDEX_ACTION = {i: act for act, i in ACTION_INDEX.items()}

ACTION_SETS = {
    'all':   ALL_ACTIONS,
    'rehab': REHAB_ACTIONS,
    'daily': DAILY_ACTIONS,
}

OL_SETS = {
    'train':    ['OL0'],
    'eval':     ['OL0', 'OL1', 'OL2'],
    'ol0':      ['OL0'],
    'ol1':      ['OL1'],
    'ol2':      ['OL2'],
    'occluded': ['OL1', 'OL2'],
}


# ──────────────────────────────────────────────────────────
#  底层 I/O
# ──────────────────────────────────────────────────────────

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1


# ── LiDAR ──────────────────────────────────────────────────

def _load_lidar_bin(path):
    """float64 XYZ bin（与 MMFi 格式相同）。"""
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)


def _load_lidar_pcd(path):
    """
    PCD 读取器。优先使用 open3d；不可用时回退到内置 ASCII/binary 解析。
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts if len(pts) > 0 else np.empty((0, 3), dtype=np.float32)
    except ImportError:
        pass

    with open(path, 'rb') as f:
        raw = f.read()

    header_end = raw.find(b'DATA ')
    if header_end == -1:
        return np.empty((0, 3), dtype=np.float32)

    header    = raw[:header_end].decode('ascii', errors='replace')
    data_type = 'ascii'
    fields, sizes, types = [], [], []
    n_points  = 0

    for line in header.split('\n'):
        tok = line.strip().split()
        if not tok:
            continue
        key = tok[0]
        if key == 'FIELDS':
            fields = tok[1:]
        elif key == 'SIZE':
            sizes = [int(x) for x in tok[1:]]
        elif key == 'TYPE':
            types = tok[1:]
        elif key == 'POINTS':
            n_points = int(tok[1])
        elif key == 'DATA':
            data_type = tok[1].strip()

    data_start = header_end + raw[header_end:].find(b'\n') + 1

    try:
        xi, yi, zi = fields.index('x'), fields.index('y'), fields.index('z')
    except ValueError:
        return np.empty((0, 3), dtype=np.float32)

    if data_type == 'ascii':
        body  = raw[data_start:].decode('ascii', errors='replace')
        lines = [l.split() for l in body.strip().split('\n') if l.strip()]
        pts   = []
        for row in lines:
            try:
                pts.append([float(row[xi]), float(row[yi]), float(row[zi])])
            except (IndexError, ValueError):
                continue
        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)

    elif data_type == 'binary':
        body     = raw[data_start:]
        row_size = sum(sizes)
        type_map = {
            ('F', 4): np.float32, ('F', 8): np.float64,
            ('I', 4): np.int32,   ('U', 4): np.uint32,
        }
        offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        pts = []
        for i in range(n_points):
            base = i * row_size
            if base + row_size > len(body):
                break
            def _read(fi):
                dt = type_map.get((types[fi], sizes[fi]))
                if dt is None:
                    return 0.0
                o = base + offsets[fi]
                return float(np.frombuffer(body[o: o + sizes[fi]], dtype=dt)[0])
            pts.append([_read(xi), _read(yi), _read(zi)])
        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)

    return np.empty((0, 3), dtype=np.float32)


def _load_lidar(path):
    return _load_lidar_pcd(path) if path.endswith('.pcd') else _load_lidar_bin(path)


# ── mmWave（TI IWR6843，复用 MMFiViPC 的解析器）──────────────

_MM_MAGIC    = bytes([2, 1, 4, 3, 6, 5, 8, 7])
_MM_HDR_FMT  = "<QIIIIIIII"
_MM_HDR_SIZE = 40


def _try_numpy_float32_mm(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return None
    if raw.size % 5 == 0:
        pc = raw.reshape(-1, 5)
        if np.all(np.abs(pc[:, :3]) < 20):
            return pc
    if raw.size % 4 == 0:
        pc = raw.reshape(-1, 4)
        if np.all(np.abs(pc[:, :3]) < 20):
            return np.hstack([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)])
    return None


def _parse_ti_uart(path):
    raw = open(path, 'rb').read()
    all_pts, pos = [], 0
    while True:
        idx = raw.find(_MM_MAGIC, pos)
        if idx == -1:
            break
        if idx + _MM_HDR_SIZE > len(raw):
            break
        total_len = struct.unpack_from('<I', raw, idx + 12)[0]
        frame = raw[idx: idx + total_len]
        if len(frame) < _MM_HDR_SIZE:
            pos = idx + _MM_HDR_SIZE
            continue
        try:
            header = struct.unpack_from(_MM_HDR_FMT, frame, 0)
        except struct.error:
            pos = idx + _MM_HDR_SIZE
            continue
        if header[0] != struct.unpack('<Q', _MM_MAGIC)[0]:
            pos = idx + _MM_HDR_SIZE
            continue
        pts, offset = [], _MM_HDR_SIZE
        for _ in range(header[7]):
            if offset + 8 > len(frame):
                break
            tlv_type, tlv_len = struct.unpack_from('<II', frame, offset)
            offset += 8
            if tlv_type == 1:
                for _ in range(tlv_len // 16):
                    if offset + 16 > len(frame):
                        break
                    x, y, z, v = struct.unpack_from('<ffff', frame, offset)
                    pts.append([x, y, z, v, 0.0])
                    offset += 16
            elif tlv_type == 7:
                for _ in range(tlv_len // 20):
                    if offset + 20 > len(frame):
                        break
                    x, y, z, v = struct.unpack_from('<ffff', frame, offset)
                    snr, _ = struct.unpack_from('<HH', frame, offset + 16)
                    pts.append([x, y, z, v, float(snr)])
                    offset += 20
            else:
                offset += tlv_len
        if pts:
            all_pts.append(np.array(pts, dtype=np.float32))
        pos = max(idx + _MM_HDR_SIZE, idx + total_len)
    return np.vstack(all_pts) if all_pts else None


def _load_mmwave(path):
    pc = _try_numpy_float32_mm(path)
    if pc is not None and len(pc) > 0:
        return pc
    pc = _parse_ti_uart(path)
    if pc is not None and len(pc) > 0:
        return pc
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size % 5 == 0 and raw64.size > 0:
        return raw64.reshape(-1, 5).astype(np.float32)
    return np.empty((0, 5), dtype=np.float32)


# ── 通用工具 ────────────────────────────────────────────────

def _normalize(pc):
    c   = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s   = max(float(np.max(np.sqrt((pc_c ** 2).sum(1)))), 1e-8)
    return (pc_c / s).astype(np.float32), c.astype(np.float32), s


def _subsample(pc, n):
    N = pc.shape[0]
    if N == 0:
        return np.zeros((n, pc.shape[-1]), dtype=pc.dtype)
    return pc[np.random.choice(N, n, replace=N < n)]


def _load_action_labels(subj_dir, n_frames, verbose):
    """
    加载 action_labels.npy（首选）或 segments.json（回退）。

    action_labels.npy : shape (N,) int，值域 0–29（见 ACTION_INDEX）
    segments.json     : {"R01": [start, end), ...}  帧索引范围（含左不含右）

    返回 np.ndarray shape (N,) dtype int32，或 None（两者均缺失时）。
    """
    label_path = os.path.join(subj_dir, 'action_labels.npy')
    if os.path.isfile(label_path):
        labels = np.load(label_path).astype(np.int32)
        if len(labels) != n_frames and verbose:
            print(f'[WARN] action_labels 长度 {len(labels)} ≠ 帧数 {n_frames}: {subj_dir}')
        return labels

    seg_path = os.path.join(subj_dir, 'segments.json')
    if os.path.isfile(seg_path):
        with open(seg_path) as f:
            segs = json.load(f)
        labels = np.full(n_frames, -1, dtype=np.int32)
        for act, (s, e) in segs.items():
            idx = ACTION_INDEX.get(act, -1)
            if idx >= 0:
                labels[s:e] = idx
        return labels

    if verbose:
        print(f'[WARN] 找不到 action_labels.npy 或 segments.json: {subj_dir}')
    return None


# ──────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────

class MISTViPCDataset(Dataset):
    """
    MIST 数据集加载器。

    每帧返回四元组 (skeleton, lidar, mmwave, label)，与 MMFiViPC 接口完全一致：
      skeleton : (17, 3)  float32  归一化后的 H36M 骨骼，(depth, lateral, height)
      lidar    : (lidar_npoints, 3)  float32  归一化 LiDAR GT
      mmwave   : (mmwave_npoints, 3) float32  独立归一化 mmWave 稀疏云
      label    : ()  int64  动作索引（0–29，见 ACTION_INDEX）
    """

    def __init__(self, data_root, ol_levels, subjects, actions,
                 split='train', lidar_npoints=2048, mmwave_npoints=512,
                 normalize=True, min_lidar_pts=32, verbose=True):
        self.lidar_npoints  = lidar_npoints
        self.mmwave_npoints = mmwave_npoints
        self.normalize      = normalize
        self.min_lidar_pts  = min_lidar_pts
        self._records = []

        wanted_indices = {ACTION_INDEX[a] for a in actions if a in ACTION_INDEX}

        for ol in sorted(ol_levels):
            for subj in sorted(subjects):
                subj_dir = os.path.join(data_root, ol, subj)
                if not os.path.isdir(subj_dir):
                    if verbose:
                        print(f'[WARN] 目录不存在: {subj_dir}')
                    continue

                # ── 读取骨骼 GT ──────────────────────────────
                skel_path = os.path.join(subj_dir, 'skeleton.npy')
                if not os.path.isfile(skel_path):
                    if verbose:
                        print(f'[WARN] skeleton.npy 缺失: {subj_dir}')
                    continue
                try:
                    gt_raw = np.load(skel_path, allow_pickle=True)
                    if gt_raw.dtype == object:
                        gt_raw = np.array(gt_raw.tolist(), dtype=np.float32)
                    else:
                        gt_raw = gt_raw.astype(np.float32)
                    sh = gt_raw.shape
                    if   gt_raw.ndim == 3 and sh[1] == 17 and sh[2] == 3:
                        gt = gt_raw
                    elif gt_raw.ndim == 3 and sh[1] == 3  and sh[2] == 17:
                        gt = gt_raw.transpose(0, 2, 1)
                    elif gt_raw.ndim == 2 and sh[1] == 51:
                        gt = gt_raw.reshape(sh[0], 17, 3)
                    else:
                        if verbose:
                            print(f'[WARN] skeleton shape 无法解析 {sh}: {skel_path}')
                        continue
                except Exception as e:
                    if verbose:
                        print(f'[WARN] skeleton 加载失败 {skel_path}: {e}')
                    continue

                n_frames = len(gt)

                # ── 读取动作标签 ──────────────────────────────
                labels = _load_action_labels(subj_dir, n_frames, verbose)
                if labels is None:
                    continue

                # ── 扫描帧文件 ────────────────────────────────
                l_files = {}
                for ext in ('*.bin', '*.pcd'):
                    for f in glob.glob(os.path.join(subj_dir, 'lidar', ext)):
                        fn = _frame_num(f)
                        if fn != -1 and fn not in l_files:
                            l_files[fn] = f

                m_files = {}
                for f in glob.glob(os.path.join(subj_dir, 'mmwave', '*.bin')):
                    fn = _frame_num(f)
                    if fn != -1:
                        m_files[fn] = f

                # frame 编号以 1 为起点，对应 skeleton 索引 fn-1
                for fn in sorted(set(l_files) & set(m_files)):
                    si = fn - 1
                    if si < 0 or si >= n_frames:
                        continue
                    act_idx = int(labels[si])
                    if act_idx not in wanted_indices:
                        continue
                    self._records.append(
                        (l_files[fn], m_files[fn], gt[si], act_idx))

        random.Random(42).shuffle(self._records)

        if verbose:
            print(f'MISTViPCDataset [{split}]: {len(self._records)} frames  '
                  f'(OL={ol_levels}, subj={subjects}, '
                  f'lidar={lidar_npoints}, mmwave={mmwave_npoints})')
        if not self._records:
            raise RuntimeError(
                f'No data loaded from {data_root}. '
                f'ol_levels={ol_levels}, subjects={subjects}, actions={actions}')

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        l_path, m_path, skel_frame, act_idx = self._records[idx]

        # ── LiDAR GT ──────────────────────────────────────────
        lidar = _load_lidar(l_path)
        if lidar.shape[0] < self.min_lidar_pts:
            lidar = np.zeros((self.lidar_npoints, 3), dtype=np.float32)
            shift = np.zeros((1, 3), dtype=np.float32)
            scale = 1.0
        elif self.normalize:
            lidar, shift, scale = _normalize(lidar)
        else:
            shift = np.zeros((1, 3), dtype=np.float32)
            scale = 1.0

        lidar = torch.from_numpy(_subsample(lidar, self.lidar_npoints)).float()

        # ── mmWave（条件输入）─────────────────────────────────
        mm = _load_mmwave(m_path)
        mm_xyz = mm[:, :3] if mm.shape[0] > 0 else np.zeros((1, 3), dtype=np.float32)

        valid = (np.isfinite(mm_xyz).all(axis=1) &
                 (np.abs(mm_xyz) < 20.0).all(axis=1))
        mm_xyz = mm_xyz[valid]
        if mm_xyz.shape[0] == 0:
            mm_xyz = np.zeros((1, 3), dtype=np.float32)

        if self.normalize and mm_xyz.shape[0] > 1:
            mm_xyz, _, _ = _normalize(mm_xyz)

        mm_xyz = torch.from_numpy(_subsample(mm_xyz, self.mmwave_npoints)).float()

        # ── Skeleton（条件输入）───────────────────────────────
        # 原始 skeleton.npy 坐标系（相机世界）：(lateral, height, depth)
        # 轴变换 → (depth, lateral, height)，与 LiDAR/mmWave 对齐
        skel = skel_frame[:, [2, 0, 1]].copy()   # (17, 3)
        if self.normalize:
            skel = ((skel - shift) / scale).astype(np.float32)

        skeleton = torch.from_numpy(skel).float()

        # ── 动作标签 ──────────────────────────────────────────
        label = torch.tensor(act_idx, dtype=torch.long)

        return skeleton, lidar, mm_xyz, label

    @property
    def pc_input_num(self):
        """兼容 ViPC 外部访问接口。"""
        return self.mmwave_npoints


# ──────────────────────────────────────────────────────────
#  与 MMFiViPC.py 完全相同的接口入口
# ──────────────────────────────────────────────────────────

def get_data_loaders(cfg):
    """
    cfg 需要的字段：
      data_dir             : str   MIST 根目录，e.g. "data/MIST"
      train_subjects       : list  e.g. ['P01','P02','P03','P04','P05','P06']
      val_subjects         : list  e.g. ['P07','P08']
      train_ol_levels      : list | str  e.g. ['OL0'] 或 'train'
      eval_ol_levels       : list | str  e.g. ['OL0','OL1','OL2'] 或 'eval'
      actions              : list  （优先）或
      action_set           : str   'all' / 'rehab' / 'daily'
      tr_max_sample_points : int   LiDAR 点数（默认 2048）
      mmwave_npoints       : int   mmWave 点数（默认 512）
      normalize            : bool
      batch_size           : int
      test_batch_size      : int
      num_workers          : int
    """
    root = cfg.data_dir

    if hasattr(cfg, 'actions') and cfg.actions:
        actions = list(cfg.actions)
    else:
        action_set = getattr(cfg, 'action_set', 'all')
        actions = ACTION_SETS.get(action_set, ALL_ACTIONS)

    def _resolve_ol(val, default_key):
        if val is None:
            return OL_SETS[default_key]
        if isinstance(val, str):
            return OL_SETS.get(val, [val])
        return list(val)

    train_ol = _resolve_ol(getattr(cfg, 'train_ol_levels', None), 'train')
    eval_ol  = _resolve_ol(getattr(cfg, 'eval_ol_levels',  None), 'eval')

    lidar_n  = getattr(cfg, 'tr_max_sample_points', 2048)
    mmwave_n = getattr(cfg, 'mmwave_npoints', 512)
    norm     = getattr(cfg, 'normalize', True)

    tr_dataset = MISTViPCDataset(
        root, train_ol, list(cfg.train_subjects), actions,
        split='train', lidar_npoints=lidar_n,
        mmwave_npoints=mmwave_n, normalize=norm)

    te_dataset = MISTViPCDataset(
        root, eval_ol, list(cfg.val_subjects), actions,
        split='val', lidar_npoints=lidar_n,
        mmwave_npoints=mmwave_n, normalize=norm)

    train_loader = tdata.DataLoader(
        tr_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=getattr(cfg, 'num_workers', 4),
        drop_last=True, pin_memory=True)

    test_loader = tdata.DataLoader(
        te_dataset, batch_size=getattr(cfg, 'test_batch_size', cfg.batch_size),
        shuffle=False, num_workers=getattr(cfg, 'num_workers', 4),
        drop_last=False)

    return {'train_loader': train_loader, 'test_loader': test_loader}


# ──────────────────────────────────────────────────────────
#  辅助工具：生成 action_labels.npy（采集后一次性运行）
# ──────────────────────────────────────────────────────────

def build_action_labels_from_segments(subj_dir, segments, n_frames, save=True):
    """
    从 segments dict 构建并保存 action_labels.npy。

    segments 格式：
        {"R01": [0, 150], "R02": [150, 310], ...}
        含左不含右，与 Python slice 语义一致。

    示例：
        build_action_labels_from_segments(
            "data/MIST/OL0/P01",
            {"R01": [0,150], "D01": [150,300]},
            n_frames=300
        )
    """
    labels = np.full(n_frames, -1, dtype=np.int32)
    for act, (s, e) in segments.items():
        idx = ACTION_INDEX.get(act, -1)
        if idx < 0:
            print(f'[WARN] 未知动作: {act}')
            continue
        labels[s:e] = idx
    if save:
        out = os.path.join(subj_dir, 'action_labels.npy')
        np.save(out, labels)
        print(f'Saved {out}  shape={labels.shape}')
    return labels


# ──────────────────────────────────────────────────────────
#  快速自测
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root',     default='data/MIST')
    p.add_argument('--ol',       nargs='+', default=['OL0'])
    p.add_argument('--subjects', nargs='+', default=['P01'])
    p.add_argument('--actions',  nargs='+', default=['R01', 'D01'])
    p.add_argument('--lidar_n',  type=int,  default=256)
    p.add_argument('--mmwave_n', type=int,  default=256)
    args = p.parse_args()

    ds = MISTViPCDataset(
        args.root, args.ol, args.subjects, args.actions,
        lidar_npoints=args.lidar_n, mmwave_npoints=args.mmwave_n)

    skeleton, lidar, mmwave, label = ds[0]

    print("=" * 60)
    print(f"skeleton : {skeleton.shape}  label={label.item()} ({INDEX_ACTION[label.item()]})")
    print(f"  depth   range: [{skeleton[:,0].min():.3f}, {skeleton[:,0].max():.3f}]")
    print(f"  lateral range: [{skeleton[:,1].min():.3f}, {skeleton[:,1].max():.3f}]")
    print(f"  height  range: [{skeleton[:,2].min():.3f}, {skeleton[:,2].max():.3f}]")
    print()
    print(f"lidar    : {lidar.shape}")
    print(f"  depth   range: [{lidar[:,0].min():.3f}, {lidar[:,0].max():.3f}]")
    print(f"  lateral range: [{lidar[:,1].min():.3f}, {lidar[:,1].max():.3f}]")
    print(f"  height  range: [{lidar[:,2].min():.3f}, {lidar[:,2].max():.3f}]")
    print()
    print(f"mmwave   : {mmwave.shape}")
    print(f"  depth   range: [{mmwave[:,0].min():.3f}, {mmwave[:,0].max():.3f}]")
    print(f"  lateral range: [{mmwave[:,1].min():.3f}, {mmwave[:,1].max():.3f}]")
    print(f"  height  range: [{mmwave[:,2].min():.3f}, {mmwave[:,2].max():.3f}]")
    print("=" * 60)
    print("验证：三种模态坐标范围应大体重叠（约 [-1.5, 1.5]）")
    print("若 skeleton 与 lidar 的 depth/height range 差异 < 0.5 则对齐成功")
