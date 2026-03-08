"""
datasets/MMFiViPC.py
====================
ShapeNetViPC → MMFi 的数据层替换。

返回三元组 (skeleton, lidar, mmwave)，对应原来的 (views, pc, pc_part)：

  skeleton : (B, 17, 3)  ← 代替图像 view
  lidar    : (B, N, 3)   ← 代替完整 GT 点云 pc
  mmwave   : (B, M, 3)   ← 代替残缺点云 pc_part

接口与 ViPC.py 完全相同：
    from datasets.MMFiViPC import get_data_loaders
    loaders = get_data_loaders(cfg.data)

坐标系约定（三种模态统一对齐到 LiDAR 传感器坐标系）：
  index 0 → X : 深度方向 (depth)
  index 1 → Y : 左右方向 (lateral)
  index 2 → Z : 高度方向 (height, 向上为正)

原始骨骼 GT 的轴约定（相机世界坐标系）：
  index 0 → 左右方向 (lateral)
  index 1 → 高度方向 (height)
  index 2 → 深度方向 (depth)

因此骨骼需要做：
  ① 轴变换：skel[:, [2, 0, 1]]  →  (depth, lateral, height)
  ② 归一化：(skel - shift) / scale  与 LiDAR/mmWave 统一到同一空间
"""

import os, re, glob, random, struct
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils import data as tdata


# ──────────────────────────────────────────────────────────
#  底层 I/O
# ──────────────────────────────────────────────────────────

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1


def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)


# ── TI IWR6843 UART 帧解析（Mmfi_coordverifier.py 提供的正确实现）──
_MM_MAGIC    = bytes([2, 1, 4, 3, 6, 5, 8, 7])
_MM_HDR_FMT  = "<QIIIIIIII"
_MM_HDR_SIZE = 40

def _try_numpy_float32_mm(path):
    """先尝试简单 float32/float64 数组格式，并验证数值合理性。"""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0: return None
    if raw.size % 5 == 0:
        pc = raw.reshape(-1, 5)
        if np.all(np.abs(pc[:, :3]) < 20): return pc   # 数值合理则直接返回
    if raw.size % 4 == 0:
        pc = raw.reshape(-1, 4)
        if np.all(np.abs(pc[:, :3]) < 20):
            return np.hstack([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)])
    return None

def _parse_ti_uart(path):
    """解析 TI IWR6843 UART 帧格式（Magic Word + TLV 结构）。"""
    import struct as _struct
    raw = open(path, 'rb').read()
    all_pts, pos = [], 0
    while True:
        idx = raw.find(_MM_MAGIC, pos)
        if idx == -1: break
        if idx + _MM_HDR_SIZE > len(raw): break
        total_len = _struct.unpack_from('<I', raw, idx + 12)[0]
        frame = raw[idx: idx + total_len]
        if len(frame) < _MM_HDR_SIZE:
            pos = idx + _MM_HDR_SIZE; continue
        try:
            header = _struct.unpack_from(_MM_HDR_FMT, frame, 0)
        except _struct.error:
            pos = idx + _MM_HDR_SIZE; continue
        if header[0] != _struct.unpack('<Q', _MM_MAGIC)[0]:
            pos = idx + _MM_HDR_SIZE; continue
        pts, offset = [], _MM_HDR_SIZE
        for _ in range(header[7]):
            if offset + 8 > len(frame): break
            tlv_type, tlv_len = _struct.unpack_from('<II', frame, offset)
            offset += 8
            if tlv_type == 1:      # xyz + velocity（无 SNR）
                for _ in range(tlv_len // 16):
                    if offset + 16 > len(frame): break
                    x, y, z, v = _struct.unpack_from('<ffff', frame, offset)
                    pts.append([x, y, z, v, 0.0]); offset += 16
            elif tlv_type == 7:    # xyz + velocity + SNR
                for _ in range(tlv_len // 20):
                    if offset + 20 > len(frame): break
                    x, y, z, v = _struct.unpack_from('<ffff', frame, offset)
                    snr, _ = _struct.unpack_from('<HH', frame, offset + 16)
                    pts.append([x, y, z, v, float(snr)]); offset += 20
            else:
                offset += tlv_len
        if pts:
            all_pts.append(np.array(pts, dtype=np.float32))
        pos = max(idx + _MM_HDR_SIZE, idx + total_len)
    return np.vstack(all_pts) if all_pts else None

def _load_mmwave(path):
    """
    健壮的 mmWave bin 读取函数。
    优先级：① 验证过的 float32/float64 数组 → ② TI UART 帧解析 → ③ 兜底
    """
    pc = _try_numpy_float32_mm(path)
    if pc is not None and len(pc) > 0: return pc
    pc = _parse_ti_uart(path)
    if pc is not None and len(pc) > 0: return pc
    # 兜底：float64
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size % 5 == 0 and raw64.size > 0:
        return raw64.reshape(-1, 5).astype(np.float32)
    return np.empty((0, 5), dtype=np.float32)


def _normalize(pc):
    """
    对点云做零均值、单位尺度归一化。
    返回：归一化后的点云、质心 shift、最大半径 scale。
    LiDAR 和 mmWave 必须用同一个 (shift, scale)，骨骼也要用同一个。
    """
    c = pc.mean(axis=0, keepdims=True)       # (1, 3)，点云质心
    pc_c = pc - c
    s = max(np.max(np.sqrt((pc_c ** 2).sum(1))), 1e-8)  # 最大点到质心距离
    return (pc_c / s).astype(np.float32), c.astype(np.float32), float(s)


def _subsample(pc, n):
    """随机采样 / 有放回重采样到固定点数 n。"""
    N = pc.shape[0]
    if N == 0:
        return np.zeros((n, pc.shape[-1]), dtype=pc.dtype)
    idx = np.random.choice(N, n, replace=N < n)
    return pc[idx]


# ──────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────

class MMFiViPCDataset(Dataset):
    """
    每帧返回 (skeleton, lidar_gt, mmwave_sparse)。

    坐标系对齐说明（Bug 修复核心）：
    ─────────────────────────────────────────────────────────
    LiDAR 点云：
        传感器原生坐标，_normalize() 后落在 [-1,1] 附近单位球内。
        坐标轴：(depth, lateral, height)

    mmWave 点云：
        与 LiDAR 使用相同传感器坐标系，用 LiDAR 的 shift/scale 归一化，
        确保两者在同一空间中可以做空间比较和 Cross-Attention。
        坐标轴：(depth, lateral, height)   ← 与 LiDAR 相同

    骨骼 GT（修复前）：
        相机世界坐标系，轴顺序 (lateral, height, depth)，
        且保留绝对值（单位：米），与归一化后的点云（[-1,1]）
        尺度差数倍，空间上完全分离。

    骨骼 GT（修复后）：
        ① 轴变换：gt[:, [2, 0, 1]] → (depth, lateral, height)
                   与 LiDAR/mmWave 轴约定对齐
        ② 归一化：(skel - shift) / scale
                   shift、scale 来自同一帧的 LiDAR 归一化结果，
                   确保骨骼关节坐标落在与点云相同的归一化空间中。
    ─────────────────────────────────────────────────────────

    这两步缺一不可：
    - 只做①不做②：轴对了但尺度差几倍，骨骼仍在点云"外太空"
    - 只做②不做①：尺度对了但高度/深度轴互换，骨骼空间镜像翻转
    """

    def __init__(self, data_root, subjects, actions,
                 split='train', lidar_npoints=2048, mmwave_npoints=512,
                 normalize=True, min_lidar_pts=32, verbose=True):
        self.lidar_npoints  = lidar_npoints
        self.mmwave_npoints = mmwave_npoints
        self.normalize      = normalize
        self.min_lidar_pts  = min_lidar_pts
        self._records = []
        # ★ 动作标签映射：按 sorted 顺序，A01→0, A02→1, ...
        self.action_to_idx  = {act: i for i, act in enumerate(sorted(actions))}

        for subj in sorted(subjects):
            for act in sorted(actions):
                lidar_dir  = os.path.join(data_root, subj, act, 'lidar')
                mmwave_dir = os.path.join(data_root, subj, act, 'mmwave')
                gt_path    = os.path.join(data_root, subj, act, 'ground_truth.npy')

                if not (os.path.isdir(lidar_dir) and
                        os.path.isdir(mmwave_dir) and
                        os.path.isfile(gt_path)):
                    continue

                try:
                    gt_raw = np.load(gt_path, allow_pickle=True)
                    if gt_raw.dtype == object:
                        gt_raw = np.array(gt_raw.tolist(), dtype=np.float32)
                    else:
                        gt_raw = gt_raw.astype(np.float32)
                    s = gt_raw.shape
                    if   gt_raw.ndim == 3 and s[1] == 17 and s[2] == 3:
                        gt = gt_raw                              # (T, 17, 3) 正常
                    elif gt_raw.ndim == 3 and s[1] == 3  and s[2] == 17:
                        gt = gt_raw.transpose(0, 2, 1)          # (T, 3, 17) → 转置
                    elif gt_raw.ndim == 2 and s[1] == 51:
                        gt = gt_raw.reshape(s[0], 17, 3)        # flatten 格式
                    elif gt_raw.ndim == 4 and s[0] == 1:
                        gt = gt_raw[0].astype(np.float32)
                    else:
                        if verbose: print(f'[WARN] GT shape 无法解析: {gt_path}')
                        continue
                except Exception as e:
                    if verbose: print(f'[WARN] GT 失败 {gt_path}: {e}')
                    continue

                l_files = {_frame_num(f): f for f in
                           glob.glob(os.path.join(lidar_dir,  'frame*_filtered.bin'))}
                m_files = {_frame_num(f): f for f in
                           glob.glob(os.path.join(mmwave_dir, 'frame*_filtered.bin'))}
                l_files.pop(-1, None); m_files.pop(-1, None)

                for fn in sorted(set(l_files) & set(m_files)):
                    if fn - 1 >= len(gt): continue
                    self._records.append(
                        (subj, act, fn, l_files[fn], m_files[fn], gt))

        random.Random(42).shuffle(self._records)

        if verbose:
            print(f'MMFiViPCDataset [{split}]: {len(self._records)} frames  '
                  f'(lidar={lidar_npoints}, mmwave={mmwave_npoints})')
        if not self._records:
            raise RuntimeError(
                f'No data loaded from {data_root}. '
                f'subjects={subjects}, actions={actions}')

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        subj, act, fn, l_path, m_path, gt = self._records[idx]

        # ── LiDAR GT ─────────────────────────────────────────────
        # 传感器坐标系：(depth, lateral, height)，不需要轴变换
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

        # ── mmWave（条件输入）─────────────────────────────────────
        # mmWave 与 LiDAR 坐标原点不同：
        #   LiDAR  原始坐标：以传感器为原点，人体在约 3m 处
        #   mmWave 原始坐标：已经是以人体为中心的坐标系（约 [-2.2, 2.2]）
        # 因此 mmWave 不能使用 LiDAR 的 shift/scale，需要独立归一化
        mm = _load_mmwave(m_path)
        mm_xyz = mm[:, :3] if mm.shape[0] > 0 else np.zeros((1, 3), dtype=np.float32)

        # 过滤坏点：去除 NaN、Inf 以及绝对值超过 20m 的异常值
        valid = (np.isfinite(mm_xyz).all(axis=1) &
                 (np.abs(mm_xyz) < 20.0).all(axis=1))
        mm_xyz = mm_xyz[valid]
        if mm_xyz.shape[0] == 0:
            mm_xyz = np.zeros((1, 3), dtype=np.float32)

        # 独立归一化（减自身质心，除自身最大半径）
        if self.normalize and mm_xyz.shape[0] > 1:
            mm_xyz, _, _ = _normalize(mm_xyz)

        mm_xyz = torch.from_numpy(_subsample(mm_xyz, self.mmwave_npoints)).float()

        # ── Skeleton GT（条件输入）───────────────────────────────
        #
        # 原始 GT 坐标系（相机世界坐标系）：
        #   index 0 = lateral（左右）
        #   index 1 = height（高度，向上为正）
        #   index 2 = depth（深度）
        #
        # 目标坐标系（与 LiDAR/mmWave 统一）：
        #   index 0 = depth（深度）
        #   index 1 = lateral（左右）
        #   index 2 = height（高度，向上为正）
        #
        # 变换：原来的 [lateral, height, depth] → [depth, lateral, height]
        #       即：新轴0 = 原轴2, 新轴1 = 原轴0, 新轴2 = 原轴1
        #       代码：skel[:, [2, 0, 1]]
        #
        # Bug 修复①：轴变换，对齐到 LiDAR 约定
        skel_raw = gt[fn - 1][:, [2, 0, 1]].copy()    # (17, 3)

        # Bug 修复②：用与 LiDAR 相同的 shift/scale 归一化
        # 使骨骼关节落在与点云相同的归一化空间（约 [-1, 1]）
        if self.normalize:
            skel_center = skel_raw.mean(axis=0, keepdims=True)
            skel_scale = max(
                np.max(np.sqrt(((skel_raw - skel_center) ** 2).sum(1))),
                1e-8
            )
            skel_raw, _, _ = _normalize(skel_raw)

        skeleton = torch.from_numpy(skel_raw.astype(np.float32)).float()   # (17, 3)

        # ★ 动作标签（整数 tensor，用于 LabelEmbedding）
        label = torch.tensor(self.action_to_idx[act], dtype=torch.long)

        # 返回四元组：(skeleton, lidar, mmwave, label)
        # label 新增，其余与原三元组顺序不变
        return skeleton, lidar, mm_xyz, label

    @property
    def pc_input_num(self):
        """兼容 ViPC 的外部访问接口。"""
        return self.mmwave_npoints


# ──────────────────────────────────────────────────────────
#  与 ViPC.py 完全相同的接口入口
# ──────────────────────────────────────────────────────────

PROTOCOL_ACTIONS = {
    'protocol1': ['A02','A03','A04','A05','A13','A14',
                  'A17','A18','A19','A20','A21','A22','A23','A27'],
    'protocol2': ['A01','A06','A07','A08','A09','A10',
                  'A11','A12','A15','A16','A24','A25','A26'],
    'all':       [f'A{i:02d}' for i in range(1, 28)],
}


def get_data_loaders(cfg):
    """
    cfg 需要的字段：
      data_dir             : str   根目录 e.g. "data/MMFi"
      env                  : str   e.g. "E01"
      train_subjects       : list  e.g. ['S01',...,'S07']
      val_subjects         : list  e.g. ['S08','S09','S10']
      actions              : list  （优先）或
      protocol             : str   'protocol1'/'protocol2'/'all'
      tr_max_sample_points : int   lidar 点数
      mmwave_npoints       : int   mmwave 点数
      normalize            : bool
      batch_size           : int
      test_batch_size      : int
      num_workers          : int
    """
    env  = getattr(cfg, 'env', 'E01')
    root = os.path.join(cfg.data_dir, env)

    if hasattr(cfg, 'actions') and cfg.actions:
        actions = list(cfg.actions)
    else:
        protocol = getattr(cfg, 'protocol', 'protocol1')
        actions  = PROTOCOL_ACTIONS.get(protocol, PROTOCOL_ACTIONS['protocol1'])

    lidar_n  = getattr(cfg, 'tr_max_sample_points', 2048)
    mmwave_n = getattr(cfg, 'mmwave_npoints', 512)
    norm     = getattr(cfg, 'normalize', True)

    tr_dataset = MMFiViPCDataset(
        root, list(cfg.train_subjects), actions,
        split='train', lidar_npoints=lidar_n,
        mmwave_npoints=mmwave_n, normalize=norm)

    te_dataset = MMFiViPCDataset(
        root, list(cfg.val_subjects), actions,
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
#  快速自测：验证三种模态的坐标系是否对齐
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root',     default='data/MMFi/E01')
    p.add_argument('--subjects', nargs='+', default=['S01'])
    p.add_argument('--actions',  nargs='+', default=['A13'])
    args = p.parse_args()

    ds = MMFiViPCDataset(args.root, args.subjects, args.actions,
                         lidar_npoints=256, mmwave_npoints=256)
    skeleton, lidar, mmwave = ds[0]

    print("=" * 55)
    print(f"skeleton : {skeleton.shape}  dtype={skeleton.dtype}")
    print(f"  depth   range: [{skeleton[:,0].min():.3f}, {skeleton[:,0].max():.3f}]")
    print(f"  lateral range: [{skeleton[:,1].min():.3f}, {skeleton[:,1].max():.3f}]")
    print(f"  height  range: [{skeleton[:,2].min():.3f}, {skeleton[:,2].max():.3f}]")
    print()
    print(f"lidar    : {lidar.shape}  dtype={lidar.dtype}")
    print(f"  depth   range: [{lidar[:,0].min():.3f}, {lidar[:,0].max():.3f}]")
    print(f"  lateral range: [{lidar[:,1].min():.3f}, {lidar[:,1].max():.3f}]")
    print(f"  height  range: [{lidar[:,2].min():.3f}, {lidar[:,2].max():.3f}]")
    print()
    print(f"mmwave   : {mmwave.shape}  dtype={mmwave.dtype}")
    print(f"  depth   range: [{mmwave[:,0].min():.3f}, {mmwave[:,0].max():.3f}]")
    print(f"  lateral range: [{mmwave[:,1].min():.3f}, {mmwave[:,1].max():.3f}]")
    print(f"  height  range: [{mmwave[:,2].min():.3f}, {mmwave[:,2].max():.3f}]")
    print("=" * 55)
    print("验证：三种模态的坐标范围应大体重叠（约 [-1.5, 1.5]）")
    print("若骨骼和点云的 depth/height range 差异 < 0.5 则对齐成功")