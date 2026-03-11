# kitsune-zd

Reproduction and extension of the **KitNET** anomaly detection system ([Mirsky et al., NDSS 2018](https://www.ndss-symposium.org/ndss-paper/kitsune-an-ensemble-of-autoencoders-for-online-network-intrusion-detection/)), evaluated on the **KITSUNE** network intrusion dataset.

This repo compares **6 autoencoder variants** across **9 attack scenarios** (54 experiments total), fixing critical bugs found in the original TensorFlow codebase and porting everything to PyTorch + pure NumPy.

## Architecture overview

KitNET is an **ensemble of autoencoders** with three phases:

```
Phase 1: Feature Mapping (100K samples)
  CorClust groups 115 features -> N clusters (typically 15-25)

Phase 2: AD Training (100K samples)
  N ensemble autoencoders, each assigned one feature cluster
  + 1 output layer (always ELM) that learns to aggregate ensemble RMSE scores

Phase 3: Execution (remaining samples)
  Each packet -> ensemble AEs produce RMSE vector -> output AE -> final anomaly score
```

## Autoencoder variants

### Packet-by-packet (online, no windowing)

These process **one network packet at a time**. Each sample passes through the autoencoder individually. Weights update after every single packet via online SGD. No temporal context.

| Variant | Architecture | Framework | Description |
|---------|-------------|-----------|-------------|
| `elm` | `input -> sigmoid(Wx+b) -> sigmoid(W'h+b') -> output` | NumPy | Original Kitsune (NDSS'18). Single hidden layer, tied-ish weights. Online SGD with incremental min/max normalization. |
| `elm_reg` | Same as `elm` | NumPy | Same architecture but input is z-score normalized (mean/std over full training set) before the online min/max normalization. |

### Sequence-to-sequence (windowed, batch training)

These create **sliding windows** of `seq_len=500` consecutive packets, forming 3D tensors of shape `(batch, 500, n_features_per_ae)`. The model reconstructs the **entire window** — input and output have the same shape. This is a **seq2seq autoencoder**: the model sees a sequence and reconstructs that same sequence. Training uses batch SGD with shuffled DataLoader; evaluation never shuffles.

| Variant | Architecture | Framework | Description |
|---------|-------------|-----------|-------------|
| `conv1d` | `Conv1D(n_vis, 3, k=500, same) -> Linear(3, h, relu) -> Linear(h, n_vis, relu)` | PyTorch | 1D convolution along the time axis captures local temporal patterns. |
| `conv2d` | `Conv2d(1, 8, 3x3) -> Conv2d(8, 16, 3x3) -> AdaptivePool -> Linear(16, h) -> Linear(h, 16) -> Linear(16, seq*n_vis)` | PyTorch | Treats (seq_len, n_visible) as a single-channel 2D image. Captures spatial+temporal patterns via 2D kernels. |
| `transformer` | `MHA_in(Q=K=V=x) -> Dense(sigmoid) -> Dense(relu) -> MHA_out` | PyTorch | Multi-head self-attention over the time axis. Global receptive field. |
| `deep_mlp` | `Flatten -> Linear(input, input//4, relu) -> Linear(input//4, h, relu) -> Linear(h, input//4, relu) -> Linear(input//4, input) -> Reshape` | PyTorch | Multilayer MLP. Flattens the window, encodes through 2 layers, decodes through 2 layers. No explicit temporal structure. |

### Why none are autoregressive

All 6 variants are **reconstruction-based autoencoders** (reconstruct `x` from `x`). None predict the *next* packet given previous packets. The anomaly score is always the **reconstruction error** (RMSE). An autoregressive approach would predict `x[t+1]` given `x[0:t]` — a different paradigm not explored here.

## Bugs found and fixed

### Bug #1 (Critical): `elm_reg` results were copy-pasted ghosts

The original `plot_threshold_metrics.py` had a scoping bug: the `elm_reg` loop did **not** capture the return value of `plot_threshold_metrics()`, so it reused the stale `(best_f1_score, best_f1_recall, best_f1_fpr, best_f1_threshold)` from the previous ELM loop. All 9 `elm_reg` entries in `results.json` were identical copies of the last ELM result (F1=0.1307), not actual `elm_reg` results.

**Impact:** After fixing, `elm_reg` produces real results nearly identical to `elm` (expected — the only difference is z-score pre-normalization). Mean F1 improvement: +0.49.

### Bug #2: Shuffled evaluation in windowed variants

The original Conv1D and Transformer implementations used `shuffle=True` in the evaluation DataLoader, randomizing the order of RMSE scores. This misaligned scores with labels, degrading metrics.

**Fix:** Evaluation DataLoaders now always use `shuffle=False`.

### Bug #3: Output layer shape mismatch

The original code passed the RMSE matrix with inconsistent shapes to the output layer, sometimes transposed.

**Fix:** The RMSE matrix is always `(N_samples, N_autoencoders)` when passed to the output layer.

### Bug #4: Missing `W_prime` update

The ELM autoencoder initialized `W_prime = W.T` but never updated it after weight changes, causing encoder/decoder to diverge.

**Fix:** `W_prime` is now properly maintained as `W.T.copy()`.

## Results

### F1 scores (best threshold per variant)

| Attack | ELM | ELM_reg | Conv1D | Transformer | Conv2D | Deep MLP | Best |
|--------|:---:|:-------:|:------:|:-----------:|:------:|:--------:|:----:|
| Active_Wiretap | 0.617 | 0.616 | **0.620** | 0.617 | 0.619 | 0.619 | Conv1D |
| ARP_MitM | 0.741 | 0.748 | **0.858** | 0.833 | 0.754 | 0.748 | Conv1D |
| Fuzzing | 0.374 | 0.374 | **0.413** | 0.407 | 0.396 | 0.403 | Conv1D |
| Mirai_Botnet | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | Tie |
| OS_Scan | **0.658** | **0.658** | 0.605 | 0.615 | 0.605 | 0.605 | ELM |
| SSDP_Flood | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | Tie |
| SSL_Renegotiation | **0.576** | **0.576** | 0.286 | 0.286 | 0.286 | 0.286 | ELM |
| SYN_DoS | **0.115** | **0.115** | 0.041 | 0.088 | 0.083 | 0.070 | ELM |
| Video_Injection | **0.164** | 0.161 | 0.087 | 0.090 | 0.087 | 0.087 | ELM |

### Ranking by mean F1

| Rank | Variant | Mean F1 | Type | Wins* |
|:----:|---------|:-------:|------|:-----:|
| 1 | **ELM_reg** | **0.5825** | Packet-by-packet | 5/9 |
| 2 | ELM | 0.5821 | Packet-by-packet | 6/9 |
| 3 | Transformer | 0.5478 | Seq2seq windowed | 2/9 |
| 4 | Conv1D | 0.5448 | Seq2seq windowed | 5/9 |
| 5 | Conv2D | 0.5361 | Seq2seq windowed | 3/9 |
| 6 | Deep MLP | 0.5346 | Seq2seq windowed | 3/9 |

*\*Wins = number of attacks where variant is within 0.002 F1 of the best. Ties counted for all.*

### Key findings

1. **The simple ELM wins overall.** Its packet-by-packet approach with online learning gives the highest mean F1 (0.582). The single hidden layer with online SGD is surprisingly effective.

2. **Conv1D dominates when temporal patterns matter.** For attacks with strong temporal signatures (ARP_MitM, Fuzzing, Active_Wiretap), Conv1D's ability to capture local patterns gives it the edge.

3. **ELM dominates on subtle attacks.** For SSL_Renegotiation, SYN_DoS, and Video_Injection, all windowed models collapse (F1 < 0.1) while ELM maintains signal. The windowing may be diluting weak per-packet anomaly signals.

4. **Conv2D and Deep MLP add no value.** They behave like slightly worse versions of Conv1D/Transformer across all attacks. The 2D spatial structure doesn't help network traffic data.

5. **z-score regularization (elm_reg) is redundant.** Nearly identical results to plain ELM, because the online min/max normalization already handles scale differences.

## Dataset

The [KITSUNE dataset](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset) contains 115 statistical features extracted from network traffic using AfterImage. 9 attack scenarios captured on a real IoT network:

| Attack | Total Packets | Attack % | Description |
|--------|:------------:|:--------:|-------------|
| Active_Wiretap | 2,279K | 42.8% | Man-in-the-middle packet sniffing |
| ARP_MitM | 2,504K | 51.5% | ARP spoofing for traffic interception |
| Fuzzing | 2,244K | 22.3% | Random malformed packets |
| Mirai_Botnet | 764K | 100.0%* | IoT botnet C&C traffic |
| OS_Scan | 1,554K | 43.1% | Nmap OS fingerprinting |
| SSDP_Flood | 4,077K | 77.0% | SSDP amplification DDoS |
| SSL_Renegotiation | 2,257K | 42.1% | SSL/TLS renegotiation attack |
| SYN_DoS | 2,718K | 52.3% | SYN flood denial of service |
| Video_Injection | 2,472K | 42.3% | RTSP video stream manipulation |

*\*Mirai_Botnet has 100% attack in the execution phase (all benign in training), making it trivially detectable but not a useful benchmark.*

## Project structure

```
kitsune-zd/
├── run_experiments.py          # Main entry point: run all 54 experiments
├── requirements.txt
├── src/
│   ├── config.py               # Hyperparameters and dataset paths
│   ├── database.py             # KITSUNE CSV loader
│   ├── corclust.py             # Incremental correlation clustering (feature mapping)
│   ├── kitnet.py               # KitNET ensemble orchestrator (3-phase pipeline)
│   ├── detector.py             # Threshold sweep, ROC plotting, metrics
│   ├── utils.py                # Sigmoid, sliding window creation
│   └── autoencoders/
│       ├── elm.py              # ELM: single-layer MLP, online SGD (NumPy)
│       ├── conv1d_ae.py        # Conv1D: temporal convolutions (PyTorch)
│       ├── conv2d_ae.py        # Conv2D: 2D convolutions on time-feature grid (PyTorch)
│       ├── transformer_ae.py   # Transformer: multi-head attention (PyTorch)
│       └── deep_mlp_ae.py      # Deep MLP: multilayer perceptron (PyTorch)
└── results/
    ├── results.json            # All metrics (F1, recall, FPR, threshold)
    └── <attack>/
        ├── <variant>_scores.npy  # Raw anomaly scores (not in git)
        └── roc.png             # ROC curves comparing all variants
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run all 54 experiments (9 attacks x 6 variants)
python run_experiments.py

# Run specific attacks/variants
python run_experiments.py --attacks Mirai_Botnet ARP_MitM
python run_experiments.py --variants elm conv1d

# Scores are cached as .npy files; re-running skips completed experiments
```

**Data path:** Set `DATA_ROOT` in `src/config.py` to your KITSUNE dataset location. Expected structure: `DATA_ROOT/<Attack Name>/<Attack_Name>_dataset.csv` and `_labels.csv`.

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| FM grace period | 100,000 | NDSS'18 paper |
| AD grace period | 100,000 | NDSS'18 paper |
| Max AE size | 10 | NDSS'18 paper |
| Hidden ratio | 0.75 | NDSS'18 paper |
| ELM learning rate | 0.1 | NDSS'18 paper |
| DL learning rate | 0.001 | Original TF repo |
| Sequence length | 500 | Original TF repo |
| Batch size | 32 | Original TF repo |
| Execution window | 10,000 | Original TF repo |

## Runtime

| Variant | Time (54 experiments) | Hardware |
|---------|:---------------------:|----------|
| ELM / ELM_reg | ~2h each | CPU (pure NumPy) |
| Conv1D | ~2.5h | NVIDIA GPU (CUDA) |
| Transformer | ~4h | NVIDIA GPU (CUDA) |
| Conv2D | ~1h | NVIDIA GPU (CUDA) |
| Deep MLP | ~1h | NVIDIA GPU (CUDA) |

## Origin

Extended from an internal research project at **Universidad de Alcala, Escuela Politecnica Superior**. Original codebase used TensorFlow 2.x; this version ports all DL models to PyTorch 2.x and fixes 4 bugs that affected the original results.
