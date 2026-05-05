# PRISM: Human Point Cloud Reconstruction via Skeleton-Guided Diffusion from mmWave Radar

<p align="center">
  <a href="https://mmwave-human.github.io/PRISM">Project Page</a> •
  <a href="#">Paper (Anonymous Submission · NeurIPS 2026)</a> •
  <a href="https://github.com/mmwave-human/PRISM">Code</a>
</p>

<p align="center">
  <img src="assets/fig/PRISM architecture overview.jpg" width="900"/>
</p>

> **PRISM** (**P**oint cloud **R**econstruction v**I**a **S**keleton-guided diffusion from **M**mWave radar) reconstructs dense, action-coherent human point clouds from sparse mmWave radar returns (≈50–150 points/frame) under clear-view and through-obstacle conditions. We also release **MIST**, the first mmWave dataset with a physical barrier between sensor and subject.

---

## Abstract

mmWave FMCW radar is an ideal modality for human sensing, offering all-weather, non-contact, and privacy-preserving perception, yet its inherent sparsity severely limits downstream body analysis and no existing dataset provides a paired quantitative benchmark for dense human point cloud reconstruction under occlusion.

We present **MIST**, the first mmWave human point cloud dataset with a physical barrier between sensor and subject, providing paired clear-view and through-obstacle captures at three controlled occlusion levels with synchronized LiDAR ground truth — spanning **8 subjects, 30 action classes, and 640k paired frames**.

We propose **PRISM**, a skeleton-guided conditional latent diffusion framework reconstructing dense human point clouds from sparse mmWave radar alone. Three conditioning streams — skeleton joint positions (SkelKD), coarse body geometry (GeoAE), and action class embeddings — guide a Dynamic Transformer to denoise under a VP-SDE framework, generating LiDAR-quality human point clouds.

On MIST, PRISM achieves **COV-CD 0.362** against 0.113 for the strongest completion baseline. Under through-obstacle attenuation (≈50 points/frame), where all baselines collapse to fixed-template outputs, PRISM maintains meaningful reconstruction.

---

## Architecture

```
mmWave PC ──→ [GeoAE] ──────────────→ Coarse C (128 pts)
    │                                         │
    └──→ [SkelKD] ──────────────→ P̂ (17 joints)
    │                                         │
    └──→ [HierVAE Enc]          [Condition Network]
              │                 (145-token K/V: GCN×17 + Geo×128)
             Z₀ ──VP-SDE──→ Z_t       │
                                  [Dynamic Transformer ×6]
                                       │
                              [HierVAE Dec] → X_f (256 pts)
```

| Module | Role |
|---|---|
| **SkelKD** | 17 learnable queries attending over per-point features → radar-aligned joint positions; trained with MPJPE + occlusion augmentation |
| **HierVAE** | 6-level hierarchical VAE encoding sparse input to 8×120 structured latent; pre-trained and frozen |
| **GeoAE** | Density-invariant mapping from sparse radar cloud to 128 stable coarse points; stable under occlusion |
| **Condition Network** | Assembles SkelKD GCN tokens (17) + GeoAE tokens (128) → 145-token K/V; action identity enters AdaLN signal |
| **Dynamic Transformer** | 6 ResidualBlocks alternating cross-attention / self-attention; AdaLN driven by timestep + action + skeleton global pool |

---

## MIST Dataset

MIST is the first mmWave human point cloud dataset providing a **paired quantitative benchmark** for through-obstacle dense human point cloud reconstruction.

**Design:**
- **Occlusion levels:** OL0 (clear view) · OL1 (18 mm wooden board) · OL2 (36 mm stacked boards)
- **Sensors:** 2× TI IWR6843 FMCW radar · Livox Mid-360 LiDAR (ground truth) · Intel RealSense D435i
- **Subjects:** 8 (P01–P08) · Train: P01–P06 · Test: P07–P08
- **Actions:** 30 classes — 16 rehabilitation (R01–R16) + 14 daily living (D01–D14)
- **Scale:** ~640k paired frames across all occlusion levels

**Expected directory structure:**
```
data/
└── MIST/
    ├── P01/
    │   ├── OL0/
    │   │   ├── D06/
    │   │   │   ├── mmwave/       # .npy point clouds (TI IWR6843)
    │   │   │   ├── lidar/        # .npy point clouds (Livox Mid-360)
    │   │   │   └── skeleton/     # .npy joint coordinates (H36M-17)
    │   │   └── ...
    │   ├── OL1/
    │   └── OL2/
    └── ...
```

To be released upon paper acceptance.

---

## MM-Fi (Cross-Dataset Transfer)

We additionally evaluate cross-dataset generalisation on [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi) (40 subjects · 27 actions · 4 environments) without retraining on MM-Fi data.

**Protocol:** E01 · S08–S10 · 8 shared action classes (A03, A12, A13, A17, A19, A22, A26, A27)

| ID | Description | ID | Description |
|---|---|---|---|
| A03 | Chest expansion | A19 | Picking up object |
| A12 | Squat | A22 | Kicking |
| A13 | Raising hand | A26 | Jumping up |
| A17 | Waving hand | A27 | Bowing |

---

## Results on MIST

| Method | Venue | Type | COV-CD ↑ | MMD-CD ↓ (×10⁻³) |
|---|---|---|---|---|
| PCN | 3DV'18 | Completion | 0.083 | 9.01 |
| PoinTr | ICCV'21 | Completion | 0.113 | 8.26 |
| SnowflakeNet | ICCV'21 | Completion | 0.079 | 10.72 |
| LAKe-Net | CVPR'22 | Completion | 0.085 | 8.69 |
| mmPoint | BMVC'23 | Radar | 0.097 | 8.91 |
| LION | NeurIPS'22 | Generative | 0.185 | 7.37 |
| TIGER | CVPR'24 | Generative | 0.209 | 7.08 |
| **PRISM (Ours)** | NeurIPS'26 | **Generative** | **0.362** | **4.91** |

Evaluated on MIST-OL0, test subjects P07–P08, all 30 action classes.

---

## Getting Started

### Installation

```bash
git clone https://github.com/mmwave-human/PRISM.git
cd PRISM

conda create -n prism python=3.8
conda activate prism

pip install -r requirements.txt

cd extern/pointnet2_ops_lib
python setup.py install
cd ../..
```

### Training

PRISM uses a **4-stage progressive training** strategy.

**Stage 1 — SkelKD pre-training**
```bash
python train_SkelKD.py --save experiments
```

**Stage 2 — HierVAE pre-training**
```bash
python train_MMFi_Compressor.py --save experiments
```

**Stage 3 — Latent diffusion training**
```bash
python train_MMFi_LDT.py --save experiments
```

**Stage 4 — Joint fine-tuning**
```bash
python train_Hybrid.py --save experiments
```

### Evaluation

```bash
python eval_ldt.py \
    --resume experiments/path/to/checkpoint.pth
```

---

## Key Configurations

| Component | Parameter | Value |
|---|---|---|
| SkelKD | local_dim / global_dim | 128 / 256 |
| HierVAE | z_dim / z_scales / levels | 20 / 8 / 6 |
| GeoAE | output points | 128 |
| Condition Network | K/V tokens | 17 + 128 = 145 |
| Dynamic Transformer | hidden / blocks / heads | 256 / 6 / 4 |
| Diffusion | train_T / sample_steps | 1000 / 200 |
| Training | lr / batch_size | 1e-4 / 16 |

---

## Repository Structure

```
PRISM/
├── datasets/
│   ├── MISTViPC.py            # MIST dataloader
│   └── MMFiViPC.py            # MM-Fi dataloader
├── model/                     # Network definitions
├── trainer/
│   ├── Hybrid_Trainer.py      # Stage 4 joint fine-tuning
│   ├── Compressor_Trainer.py  # HierVAE trainer
│   └── Latent_SDE_Trainer.py  # Diffusion trainer
├── completion_trainer/        # Stage-specific trainers
├── diffusion/                 # VP-SDE implementation
├── evaluation/                # COV-CD / MMD-CD metrics
├── tools/                     # Logging and I/O utilities
├── train_SkelKD.py            # Stage 1
├── train_MMFi_Compressor.py   # Stage 2
├── train_MMFi_LDT.py          # Stage 3
├── train_Hybrid.py            # Stage 4
├── eval_ldt.py                # Evaluation
└── index.html                 # Project page
```

---

## Citation

```bibtex
@inproceedings{prism2026,
  title     = {PRISM: Human Point Cloud Reconstruction via
               Skeleton-Guided Diffusion from MmWave Radar},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## Acknowledgements

This work builds upon [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi), [LION](https://github.com/nv-tlabs/LION), and the PointNet++ operators. We thank the authors for their excellent open-source contributions.
