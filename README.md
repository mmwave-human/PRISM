# SKD-Net: Skeleton-Guided Latent Diffusion for Human Point Cloud Completion from Sparse mmWave Radar

<p align="center">
  <a href="https://skd-net.github.io/SKD-Net">Project Page</a> •
  <a href="#">Paper (Anonymous)</a> •
  <a href="https://github.com/skd-net/SKD-Net">Code</a>
</p>

<p align="center">
  <img src="assets/SKD-Net_framework.jpg" width="900"/>
</p>

> **SKD-Net** proposes a skeleton-guided latent diffusion framework for completing dense human point clouds from sparse mmWave radar observations (50–150 points/frame). By leveraging cross-modal skeleton alignment and a three-path conditional diffusion network, SKD-Net recovers topologically correct, temporally consistent human point clouds from extremely sparse radar inputs.

---

## 📋 Abstract

mmWave radar enables non-intrusive, all-weather human sensing, but its inherent sparsity (50–150 points per frame) severely limits downstream pose estimation and action recognition tasks. Existing point cloud completion methods are designed for dense inputs of rigid objects and fail under the extreme sparsity and non-rigid topology of human bodies.

We propose **SKD-Net**, a skeleton-guided latent diffusion framework for human point cloud completion from sparse mmWave radar. SKD-Net consists of two parallel paths: (①) **SkelKD** refines cross-modal skeleton joint coordinates from mmWave observations via attention-based soft aggregation; a **PE Encoder** compresses the sparse point cloud into skeleton-aligned latent tokens Z₀. (②) A **Condition Network** fuses three conditioning signals — skeleton GCN topology (17 tokens), skeleton-guided mmWave aggregation (17 tokens), and action category embedding — into a unified K/V sequence, which guides a **Dynamic Transformer** to denoise latent tokens and reconstruct a complete human point cloud via a learned **Decoder**.

Experiments on MM-Fi demonstrate that SKD-Net significantly outperforms general completion baselines (PCN, PoinTr, Snowflake) and generative baselines (LION, TIGER) on MMD-CD, COV-CD, and 1-NN-CD metrics, while producing temporally consistent human point cloud sequences.

---

## 🏗️ Framework

<p align="center">
  <img src="assets/SKD-Net_framework.jpg" width="900"/>
</p>

The framework consists of two stages:

- **① Skeleton Alignment**: SkelKD refines skeleton joints from mmWave point clouds. PE Encoder compresses the sparse input into structured latent tokens via skeleton-anchored cross-attention. Auto-encoder E generates coarse geometric points from the raw mmWave input.
- **② Completion Refinement**: Forward diffusion gradually adds noise to Z₀. The Dynamic Transformer denoises Z_T back to Z₀ conditioned on skeleton GCN features, coarse geometry tokens, and action category label. The Decoder reconstructs the final complete point cloud x_f.

---

## 🗂️ Dataset

### MM-Fi

We use the [MM-Fi dataset](https://ntu-aiot-lab.github.io/mm-fi) as our primary benchmark. MM-Fi provides synchronized multi-modal human sensing data including mmWave radar point clouds, LiDAR point clouds, and 3D skeleton joint annotations.

**Download**: Apply for access at the [official MM-Fi page](https://ntu-aiot-lab.github.io/mm-fi).

**Experimental protocol used in this work**:

| Split | Subjects | Actions |
|-------|----------|---------|
| Train | S01–S07  | A03, A12, A13, A17, A19, A22, A26, A27 |
| Val   | S08–S10  | A03, A12, A13, A17, A19, A22, A26, A27 |

**Action descriptions**:

| ID  | Description            | ID  | Description       |
|-----|------------------------|-----|-------------------|
| A03 | Chest expansion (vertical) | A19 | Picking up things |
| A12 | Squat                  | A22 | Kicking (left)    |
| A13 | Raising hand (left)    | A26 | Jumping up        |
| A17 | Waving hand (left)     | A27 | Bowing            |

**Expected directory structure**:
```
data/
└── MMFi/
    └── E01/
        ├── S01/
        │   ├── A03/
        │   │   ├── mmwave/       # raw .bin files (TI IWR6843 UART format)
        │   │   ├── lidar/        # .npy point clouds
        │   │   └── skeleton/     # .npy joint coordinates (17 joints, COCO format)
        │   └── ...
        └── ...
```

### OccluRadar-Human *(Coming Soon)*

A self-collected dataset featuring paired complete/occluded mmWave point clouds with synchronized 3D skeleton annotations, designed to evaluate completion performance under real-world occlusion scenarios. Details to be released upon paper acceptance.

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/skd-net/SKD-Net.git
cd SKD-Net

# Create conda environment
conda create -n SKD-Net python=3.8
conda activate SKD-Net

# Install dependencies
pip install -r requirements.txt

# Install pointnet2 ops
cd pointnet2_ops_lib
python setup.py install
cd ..
```

### Training

SKD-Net follows a **4-stage progressive training** strategy:

**Stage 1: SkelKD Pre-training**
```bash
python train_SkelKD.py --save experiments
```

**Stage 2: Compressor Pre-training**
```bash
python train_Compressor.py --save experiments
```

**Stage 3: Latent Diffusion Training**
```bash
python train_MMFi_LDT.py --save experiments
```

**Stage 4: Joint Fine-tuning** *(coming soon)*

### Evaluation

```bash
python train_MMFi_LDT.py \
    --save experiments \
    --evaluate True \
    --resume True \
    --resume_epoch 1000 \
    --run_dir experiments/Latent_Diffusion_Trainer/mmfi_YYYYMMDDHHMM
```

---

## 📊 Results

> ⚠️ Full quantitative results will be updated upon experiment completion.

### Point Cloud Completion on MM-Fi

Evaluation metrics follow [LION](https://github.com/nv-tlabs/LION): MMD-CD ↓, COV-CD ↑, 1-NN-CD ↓ (lower 1-NN-CD is better, ideal = 50%).

| Method | Venue | MMD-CD ↓ | COV-CD ↑ | 1-NN-CD ↓ |
|--------|-------|----------|----------|-----------|
| PCN*   | 3DV'18 | — | — | — |
| PoinTr* | ICCV'21 | — | — | — |
| Snowflake* | ICCV'21 | — | — | — |
| LAKe-Net* | CVPR'22 | — | — | — |
| LION*  | NeurIPS'22 | — | — | — |
| TIGER* | CVPR'24 | — | — | — |
| **SKD-Net (Ours)** | — | **—** | **—** | **—** |

*\* Adapted to mmWave 256-point input.*

---

## 📁 Repository Structure

```
SKD-Net/
├── datasets/
│   └── MMFiViPC.py           # MM-Fi dataloader (mmWave + LiDAR + Skeleton)
├── model/
│   ├── SkelKD/
│   │   └── Network.py        # SkelKD cross-modal skeleton detector
│   ├── Compressor/
│   │   └── Network.py        # PE Encoder + Decoder (VAE)
│   ├── AutoEncoderE/
│   │   └── Network.py        # Auto-encoder E (mmWave → coarse points)
│   └── scorenet/
│       └── score.py          # Condition Network + Dynamic Transformer
├── completion_trainer/
│   ├── SkelKD_Trainer.py
│   ├── Compressor_Trainer.py
│   └── Latent_SDE_Trainer.py
├── train_SkelKD.py
├── train_Compressor.py
├── train_MMFi_LDT.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Key Configurations

Key hyperparameters (see `experiments/*/config.yaml` for full details):

| Component | Parameter | Value |
|-----------|-----------|-------|
| SkelKD | local_dim / global_dim | 128 / 256 |
| Compressor | z_dim / z_scales | 20 / 8 |
| Score Network | hidden_size / num_blocks | 256 / 6 |
| Diffusion | train_N / sample_N | 1000 / 200 |
| Training | lr / batch_size | 1e-4 / 16 |

---

## 📝 Citation

```bibtex
@inproceedings{skdnet2025,
  title     = {SKD-Net: Skeleton-Guided Latent Diffusion for Human Point Cloud Completion from Sparse mmWave Radar},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

---

## 🙏 Acknowledgements

This work builds upon [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi), [LION](https://github.com/nv-tlabs/LION), and [LDT](https://github.com/ZhaoyangLyu/LatentDiffusionTransformer). We thank the authors for their excellent open-source contributions.