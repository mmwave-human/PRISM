"""
dataset_mmfi.py
===============
MM-Fi dataset adapter for mmPoint.

Each frame returns a dict:
  'points' : (lidar_npoints, 3)  — normalized LiDAR dense GT
  'image'  : (mmwave_npoints, 3) — normalized mmWave sparse point cloud

The dict keys mirror the original HuPRDataset so train_mmfi.py can
reuse the same data-loading pattern without touching the training logic.

Data layout expected on disk:
  <data_root>/<subject>/<action>/lidar/frame*_filtered.bin   — float64 XYZ
  <data_root>/<subject>/<action>/mmwave/frame*_filtered.bin  — TI IWR6843 format

Example:
  /home/hjc/Desktop/SKD-Net/data/MMFi/E01/S01/A02/lidar/frame1_filtered.bin
"""

import os
import re
import glob
import random
import struct

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Frame number helper ────────────────────────────────────────────────────────

def _frame_num(fname):
    m = re.search(r'frame(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1


# ── LiDAR loader ──────────────────────────────────────────────────────────────

def _load_lidar(path):
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return raw[:(raw.size // 3) * 3].reshape(-1, 3).astype(np.float32)


# ── mmWave loader (TI IWR6843 UART format, same as MMFiViPCDataset) ──────────

_MM_MAGIC   = bytes([2, 1, 4, 3, 6, 5, 8, 7])
_MM_HDR_FMT = "<QIIIIIIII"
_MM_HDR_SIZE = 40


def _try_numpy_float32_mm(path):
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
            if tlv_type == 1:          # XYZ + velocity (no SNR)
                for _ in range(tlv_len // 16):
                    if offset + 16 > len(frame):
                        break
                    x, y, z, _ = struct.unpack_from('<ffff', frame, offset)
                    pts.append([x, y, z])
                    offset += 16
            elif tlv_type == 7:        # XYZ + velocity + SNR
                for _ in range(tlv_len // 20):
                    if offset + 20 > len(frame):
                        break
                    x, y, z, _ = struct.unpack_from('<ffff', frame, offset)
                    pts.append([x, y, z])
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
        return raw64.reshape(-1, 5)[:, :3].astype(np.float32)
    return np.zeros((1, 3), dtype=np.float32)


# ── Normalization & sampling ───────────────────────────────────────────────────

def _normalize(pc):
    """Zero-mean, unit-ball normalization."""
    c = pc.mean(axis=0, keepdims=True)
    pc_c = pc - c
    s = max(float(np.max(np.sqrt((pc_c ** 2).sum(1)))), 1e-8)
    return (pc_c / s).astype(np.float32)


def _subsample(pc, n):
    """Random subsample / oversample to exactly n points."""
    N = pc.shape[0]
    if N == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = np.random.choice(N, n, replace=(N < n))
    return pc[idx]


# ── Protocol definitions (same as SKD-Net) ────────────────────────────────────

PROTOCOL_ACTIONS = {
    'protocol1': ['A02', 'A03', 'A04', 'A05', 'A13', 'A14',
                  'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27'],
    'protocol2': ['A01', 'A06', 'A07', 'A08', 'A09', 'A10',
                  'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26'],
    'all':       [f'A{i:02d}' for i in range(1, 28)],
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class MMFiDataset(Dataset):
    """
    MM-Fi dataset for mmPoint baseline training.

    Args:
        data_root      : path to environment root, e.g. '.../MMFi/E01'
        subjects       : list of subject IDs, e.g. ['S01', ..., 'S07']
        actions        : list of action IDs,  e.g. ['A02', 'A13', ...]
        lidar_npoints  : number of LiDAR GT points to sample  (default 2048)
        mmwave_npoints : number of mmWave input points        (default 512)
        min_lidar_pts  : minimum valid LiDAR points; frames with fewer are
                         replaced with zeros                  (default 32)
    """

    def __init__(self, data_root, subjects, actions,
                 lidar_npoints=2048, mmwave_npoints=512, min_lidar_pts=32,
                 cache_root=None):
        self.lidar_npoints  = lidar_npoints
        self.mmwave_npoints = mmwave_npoints
        self.min_lidar_pts  = min_lidar_pts
        self._data_root  = os.path.abspath(data_root)
        self._cache_root = os.path.abspath(cache_root) if cache_root else None
        self._records = []

        for subj in sorted(subjects):
            for act in sorted(actions):
                lidar_dir  = os.path.join(data_root, subj, act, 'lidar')
                mmwave_dir = os.path.join(data_root, subj, act, 'mmwave')
                if not (os.path.isdir(lidar_dir) and os.path.isdir(mmwave_dir)):
                    continue

                l_files = {_frame_num(f): f for f in
                           glob.glob(os.path.join(lidar_dir,  'frame*_filtered.bin'))}
                m_files = {_frame_num(f): f for f in
                           glob.glob(os.path.join(mmwave_dir, 'frame*_filtered.bin'))}
                l_files.pop(-1, None)
                m_files.pop(-1, None)

                for fn in sorted(set(l_files) & set(m_files)):
                    self._records.append((l_files[fn], m_files[fn]))

        random.Random(42).shuffle(self._records)
        print(f'MMFiDataset: {len(self._records)} frames '
              f'(lidar={lidar_npoints}, mmwave={mmwave_npoints})')

        if not self._records:
            raise RuntimeError(
                f'No data found in {data_root} for subjects={subjects}')

    def __len__(self):
        return len(self._records)

    def _npy_path(self, bin_path):
        """Return .npy cache path for a .bin file, or None if no cache_root."""
        if self._cache_root is None:
            return None
        rel = os.path.relpath(bin_path, self._data_root)
        return os.path.join(self._cache_root, os.path.splitext(rel)[0] + '.npy')

    def _load_npy_or_bin(self, bin_path, loader_fn):
        """Load from .npy cache if available, else parse the .bin file."""
        npy_path = self._npy_path(bin_path)
        if npy_path and os.path.exists(npy_path):
            return np.load(npy_path)
        return loader_fn(bin_path)

    def __getitem__(self, idx):
        l_path, m_path = self._records[idx]

        # ── LiDAR dense GT ────────────────────────────────────────────────
        lidar = self._load_npy_or_bin(l_path, _load_lidar)
        if lidar.shape[0] < self.min_lidar_pts:
            lidar = np.zeros((self.lidar_npoints, 3), dtype=np.float32)
        else:
            lidar = _normalize(lidar)
        lidar = _subsample(lidar, self.lidar_npoints)

        # ── mmWave sparse input ───────────────────────────────────────────
        mm = self._load_npy_or_bin(m_path, _load_mmwave)
        # filter outliers
        valid = (np.isfinite(mm).all(axis=1) &
                 (np.abs(mm) < 20.0).all(axis=1))
        mm = mm[valid]
        if mm.shape[0] == 0:
            mm = np.zeros((1, 3), dtype=np.float32)
        mm = _normalize(mm)
        mm = _subsample(mm, self.mmwave_npoints)

        return {
            'points': torch.from_numpy(lidar).float(),   # (lidar_npoints, 3)
            'image':  torch.from_numpy(mm).float(),       # (mmwave_npoints, 3)
        }
