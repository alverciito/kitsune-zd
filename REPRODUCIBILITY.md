# Reproducibility Guide for Kitsune-ZD

This document provides all information necessary to reproduce the experiments in the
Kitsune-ZD paper. It covers hyperparameters, model architectures across both backends
(PyTorch and TensorFlow), detector algorithms, dataset setup, and experiment execution.

---

## 1. Paper Reference

Kitsune-ZD extends the original Kitsune NIDS (NDSS 2018) with multiple deep-learning
autoencoder architectures, two operational paradigms (Time-Series Reconstruction and
Autoregressive), and two anomaly detectors (Centroid KMD and Distribution MVD). The
study evaluates 12 model variants across the KNAD dataset (9 attacks), CIC-IDS-2017,
CIC-IDS-2018, and ACI-IoT-2023.

---

## 2. Environment and Dependencies

### 2.1 PyTorch Backend (default)

File: `requirements.txt`

```
numpy>=1.24
pandas>=2.0
torch>=2.0
matplotlib>=3.7
scipy>=1.10
scikit-learn>=1.2
```

### 2.2 TensorFlow Backend

File: `requirements-tf.txt` (install in addition to base requirements)

```
tensorflow>=2.13
```

### 2.3 Installation

```bash
# PyTorch backend only
pip install -r requirements.txt

# Both backends
pip install -r requirements.txt -r requirements-tf.txt
```

### 2.4 Hardware

- `DEVICE = "cuda"` by default. Falls back to CPU if CUDA is unavailable.
- Random seed: `SEED = 1234` (set via `torch.manual_seed(1234)` or `tf.random.set_seed(1234)`).

---

## 3. Complete Hyperparameter Table (Table II)

All hyperparameters are defined in `src/common/config.py` and shared across both backends.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| FM Grace Period | `FM_GRACE_PERIOD` | 50,000 | Clustering/feature-mapping train phase (packets) |
| AD Grace Period | `AD_GRACE_PERIOD` | 150,000 | Autoencoder train phase (packets) |
| Max AE Size | `MAX_AE_SIZE` | 4 | Number of ensemble autoencoders (4 + 1 output) |
| Hidden Ratio | `HIDDEN_RATIO` | 0.22 | 78% compression (keep 22% of input dimensions) |
| Learning Rate | `LEARNING_RATE` | 0.001 | Adam optimizer LR for all models |
| Sequence Length | `SEQUENCE_LENGTH` | 800 | Context window size (packets) for windowed models |
| Execution Window | `EXECUTION_WINDOW` | 400,000 | Execution batch size (packets) |
| Batch Size | `BATCH_SIZE` | 32 | Mini-batch size for training and evaluation |
| Clustering | `DEFAULT_CLUSTERING` | `dbscan` | Feature mapping algorithm |
| Output AE | `DEFAULT_OUTPUT_AE` | `stat` | Output aggregation layer type |
| N Features | `N_FEATURES` | 115 | AfterImage feature extractor output dimensionality |
| Epsilon | `EPSILON` | 1e-16 | Numerical stability constant |
| Seed | `SEED` | 1234 | Global random seed |
| Adam Betas | -- | (0.9, 0.999) | Adam optimizer momentum parameters |
| Training Epochs | -- | 1 | All windowed models train for a single epoch |
| Eval Batch Size | -- | 128 | `BATCH_SIZE * 4` used during evaluation |

### Normalization

All models normalize input to [0, 1] using per-feature min-max:

```
x_norm = (x - x_min) / (x_max - x_min + epsilon)
```

The ELM updates normalization bounds online (sample-by-sample); all other models compute
bounds over the full training batch.

---

## 4. Architecture Details

Each autoencoder is implemented in both PyTorch (`src/torch/autoencoders/`) and
TensorFlow (`src/tf/autoencoders/`). The ELM and Statistical models are
framework-agnostic NumPy implementations in `src/common/autoencoders/`.

Notation: `n_visible` = input feature dimension per ensemble member,
`n_hidden = max(1, ceil(n_visible * hidden_ratio))`, `seq_len` = sequence length.

### 4.1 ELM (Extreme Learning Machine) -- `src/common/autoencoders/elm.py`

Backend: Pure NumPy (shared by both backends). This is the original Kitsune (NDSS 2018) autoencoder.

**Architecture:**
```
Input (n_visible)
  -> Linear(n_visible, n_hidden) + sigmoid       [Encoder]
  -> Linear(n_hidden, n_visible) + sigmoid        [Decoder]
Output (n_visible)
```

**Training:** Online SGD, one sample at a time. Weight update rule:
```
L_h2 = x_norm - z                              (reconstruction error)
L_h1 = (L_h2 @ W) * y * (1 - y)               (backprop through sigmoid)
L_W  = outer(tilde_x, L_h1) + outer(L_h2, y)  (gradient)
W   += lr * L_W
W'   = W.T                                     (tied weights, kept in sync)
h_bias += lr * L_h1
v_bias += lr * L_h2
```

**Initialization:** Xavier-uniform: `U(-1/n_visible, 1/n_visible)`.

**Loss:** Per-sample RMSE = `sqrt(mean((x_norm - z)^2))`.

**Note:** Also used as the output aggregation layer for all KitNET variants.

---

### 4.2 Statistical Anomaly Detector -- `src/common/autoencoders/statistical_ae.py`

Backend: Pure NumPy (shared by both backends).

**Architecture:** No neural network.

**Training:** Compute per-feature mean (mu) and standard deviation (sigma) over
normalized training data.

**Scoring (Eq. 3):**
```
s(x) = sum( |x_norm - mu| / (sigma + epsilon) )
```

Returns a scalar anomaly score per sample.

---

### 4.3 Conv1D Autoencoder

#### PyTorch -- `src/torch/autoencoders/conv1d_ae.py`

**Architecture (full sequence reconstruction):**
```
Input: (batch, seq_len, n_visible)
  -> Permute to (batch, n_visible, seq_len)
  -> Conv1d(in=n_visible, out=3, kernel_size=seq_len, padding='same') + ReLU
  -> Permute to (batch, seq_len, 3)
  -> Linear(3, n_hidden) + ReLU
  -> Linear(n_hidden, n_visible) + ReLU
Output: (batch, seq_len, n_visible)
```

**Loss:** MSE over full reconstructed window. Metric: per-window RMSE = `sqrt(mean((x - z)^2))` averaged over `(seq_len, n_visible)`.

#### TensorFlow -- `src/tf/autoencoders/conv1d_ae.py`

**Architecture (last-frame prediction):**
```
Encoder:
  Input: (batch, seq_len, n_visible)
  -> Conv1D(filters=n_visible, kernel_size=seq_len//4, padding='same', tanh)
  -> MaxPooling1D(pool_size=2, padding='same')
  -> Conv1D(filters=n_hidden, kernel_size=seq_len//4, padding='same', relu)
  -> GlobalAveragePooling1D()
  -> Dense(n_hidden, relu)

Decoder:
  -> Dense(n_hidden, relu)
  -> Dense(n_visible, sigmoid)
Output: (batch, n_visible)
```

**Target:** Last frame of window `x[seq_len-1]` (TSR mode) or next frame `x[seq_len]` (AR mode).

**Key difference vs PyTorch:** TF version uses a deeper encoder with pooling and predicts only the last/next frame; PyTorch version reconstructs the entire window.

---

### 4.4 Conv2D Autoencoder

#### PyTorch -- `src/torch/autoencoders/conv2d_ae.py`

**Architecture (full sequence reconstruction):**
```
Input: (batch, seq_len, n_visible)
  -> Unsqueeze to (batch, 1, seq_len, n_visible)

Encoder:
  -> Conv2d(1, 8, kernel_size=3, padding=1) + ReLU
  -> Conv2d(8, 16, kernel_size=3, padding=1) + ReLU
  -> AdaptiveAvgPool2d(1)                            -> (batch, 16)
  -> Linear(16, n_hidden) + ReLU

Decoder:
  -> Linear(n_hidden, 16) + ReLU
  -> Linear(16, seq_len * n_visible)
  -> Reshape to (batch, seq_len, n_visible)
Output: (batch, seq_len, n_visible)
```

#### TensorFlow -- `src/tf/autoencoders/conv2d_ae.py`

**Architecture (last-frame prediction):**
```
Encoder:
  Input: (batch, seq_len, n_visible)
  -> Reshape to (batch, seq_len, n_visible, 1)
  -> Conv2D(filters=32, kernel_size=(3,3), padding='same', tanh)
  -> MaxPooling2D(pool_size=(2,2), padding='same')
  -> Conv2D(filters=n_hidden, kernel_size=(3,3), padding='same', relu)
  -> GlobalAveragePooling2D()
  -> Dense(n_hidden, relu)

Decoder:
  -> Dense(n_hidden, relu)
  -> Dense(n_visible, sigmoid)
Output: (batch, n_visible)
```

**Key difference:** TF uses 32 initial filters + MaxPooling + last-frame target; PyTorch uses 8->16 filters + AdaptiveAvgPool + full-window reconstruction.

---

### 4.5 Transformer (Multi-Head Attention) Autoencoder

#### PyTorch -- `src/torch/autoencoders/transformer_ae.py`

**Architecture (full sequence reconstruction):**
```
Input: (batch, seq_len, n_visible)
  -> MultiheadAttention(embed_dim=n_visible, num_heads=H, batch_first=True)
     where H = largest divisor of n_visible <= n_hidden
  -> Linear(n_visible, n_hidden) + Sigmoid
  -> Linear(n_hidden, n_visible) + ReLU
  -> MultiheadAttention(embed_dim=n_visible, num_heads=H, batch_first=True)
Output: (batch, seq_len, n_visible)
```

Flow: `MHA_in(Q=K=V=x) -> Dense(sigmoid) -> Dense(relu) -> MHA_out(Q=K=V=x)`

#### TensorFlow -- `src/tf/autoencoders/transformer_ae.py`

**Architecture (last-frame prediction, Functional API with residual connections):**
```
Encoder:
  Input: (batch, seq_len, n_visible)
  -> MultiHeadAttention(num_heads=n_hidden, key_dim=n_visible)
  -> BatchNormalization(attention_output + input)     [residual]
  -> Dense(n_visible, tanh)
  -> BatchNormalization(dense_output + norm1)         [residual]
  -> GlobalAveragePooling1D()
  -> Dense(n_hidden, relu)

Decoder:
  -> Dense(n_hidden, relu)
  -> Dense(n_visible, sigmoid)
Output: (batch, n_visible)
```

**Key difference:** TF version includes residual connections, BatchNormalization,
GlobalAveragePooling, and last-frame prediction. PyTorch version is a simpler
symmetric MHA-Dense-MHA block with full-window reconstruction.

---

### 4.6 Deep MLP Autoencoder

#### PyTorch -- `src/torch/autoencoders/deep_mlp_ae.py`

**Architecture (full sequence reconstruction):**
```
h1 = max(1, (seq_len * n_visible) // 4)
h2 = n_hidden

Input: (batch, seq_len, n_visible) -> Flatten to (batch, seq_len * n_visible)

Encoder:
  -> Linear(seq_len * n_visible, h1) + ReLU
  -> Linear(h1, h2) + ReLU

Decoder:
  -> Linear(h2, h1) + ReLU
  -> Linear(h1, seq_len * n_visible)            [no activation]
  -> Reshape to (batch, seq_len, n_visible)
Output: (batch, seq_len, n_visible)
```

#### TensorFlow -- `src/tf/autoencoders/deep_mlp_ae.py`

**Architecture (last-frame prediction):**
```
h1 = max(1, (seq_len * n_visible) // 4)

Input: (batch, seq_len, n_visible) -> Flatten

Encoder:
  -> Dense(h1, relu)
  -> Dense(n_hidden, relu)

Decoder:
  -> Dense(h1, relu)
  -> Dense(n_visible, sigmoid)
Output: (batch, n_visible)
```

**Key difference:** TF decoder outputs `n_visible` (last frame) with sigmoid; PyTorch decoder outputs `seq_len * n_visible` (full window) with no final activation.

---

### 4.7 LSTM Autoencoder

#### PyTorch -- `src/torch/autoencoders/lstm_ae.py`

**Architecture (last-frame prediction):**
```
Input: (batch, seq_len, n_visible)

Encoder:
  -> LSTM(input_size=n_visible, hidden_size=n_hidden, batch_first=True)
  -> Take final hidden state h_n: (batch, n_hidden)

Decoder:
  -> Linear(n_hidden, n_hidden) + ReLU
  -> Linear(n_hidden, n_visible) + Sigmoid
Output: (batch, n_visible)
```

Supports autoregressive mode (`ar=True`): predict next frame `x[seq_len]` instead of
last frame `x[seq_len-1]`.

#### TensorFlow -- `src/tf/autoencoders/lstm_ae.py`

**Architecture (last-frame prediction):**
```
Encoder:
  Input: (batch, seq_len, n_visible)
  -> LSTM(units=n_hidden, return_sequences=False)

Decoder:
  -> Dense(n_hidden, relu)
  -> Dense(n_visible, sigmoid)
Output: (batch, n_visible)
```

**Note:** LSTM is architecturally identical between backends (both predict last/next frame).

---

### 4.8 Architecture Summary Table

| Model | Backend | Reconstruction | Output Activation | Windowed |
|---|---|---|---|---|
| ELM | NumPy | Full vector | Sigmoid | No (packet-by-packet) |
| Statistical | NumPy | N/A (statistical) | N/A | No (packet-by-packet) |
| Conv1D | PyTorch | Full window | ReLU | Yes |
| Conv1D | TF | Last/next frame | Sigmoid | Yes |
| Conv2D | PyTorch | Full window | None (linear) | Yes |
| Conv2D | TF | Last/next frame | Sigmoid | Yes |
| Transformer | PyTorch | Full window | MHA output | Yes |
| Transformer | TF | Last/next frame | Sigmoid | Yes |
| Deep MLP | PyTorch | Full window | None (linear) | Yes |
| Deep MLP | TF | Last/next frame | Sigmoid | Yes |
| LSTM | PyTorch | Last/next frame | Sigmoid | Yes |
| LSTM | TF | Last/next frame | Sigmoid | Yes |

---

## 5. Anomaly Detectors

### 5.1 Threshold Sweep (Default)

The default detector sweeps over log-transformed anomaly scores to find the threshold
maximizing F1 score:

```
prediction(t) = 1  if  log(score(t) + 1e-9) >= threshold
                0  otherwise
```

### 5.2 Centroid Detector (KMD) -- `src/common/detectors/centroid.py`

From IEEE paper Section IV-D.

**Training:** K-means++ clustering on the Detection Frame scores.

```
Parameters:
  n_clusters = 8 (K)
  n_init = 10
  init = 'k-means++'
  random_state = 1234
  filter_window = 100 (mean filter)
```

**Execution:** Anomaly score is the distance to the nearest centroid:

```
D(x, t) = min_k |score(t) - centroid_k|
```

Post-processed with a sliding-window mean filter (window=100).

### 5.3 Distribution Detector (MVD) -- `src/common/detectors/distribution.py`

From IEEE paper Equation 4.

**Training:** Compute mean (mu_T) and standard deviation (sigma_T) over the Detection Frame.

**Execution:** For each time step, compute local statistics over a sliding window (W=10,000):

```
mu_d    = mean(scores[max(0, t-W+1) : t+1])
sigma_d = std(scores[max(0, t-W+1) : t+1])
```

Harmonic-mean deviation score:

```
D(x, t) = 2 * (mu_T - mu_d)^2 * (sigma_T - sigma_d)^2
           / ((mu_T - mu_d)^2 + (sigma_T - sigma_d)^2 + epsilon)
```

Post-processed with a sliding-window median filter (window=100).

```
Parameters:
  window_size = 10,000
  filter_window = 100 (median filter)
```

---

## 6. Datasets

### 6.1 KNAD (Kitsune Network Attack Dataset)

Default dataset. Root path: `/mnt/hdd8tb/database/KITSUNE`

Each attack is a subdirectory containing AfterImage feature CSVs (115 features).

**Attacks used in the paper (Table I, Fig. 7):**

| Attack | Directory Name |
|---|---|
| SSDP_Flood | `SSDP Flood` |
| Active_Wiretap | `Active Wiretap` |
| Fuzzing | `Fuzzing` |
| OS_Scan | `OS Scan` |
| Video_Injection | `Video Injection` |
| SSL_Renegotiation | `SSL Renegotiation` |

**Additional attacks available in the codebase:**

| Attack | Directory Name |
|---|---|
| ARP_MitM | `ARP MitM` |
| Mirai_Botnet | `Mirai Botnet` |
| SYN_DoS | `SYN DoS` |

### 6.2 CIC-IDS-2017

Root path: `/mnt/hdd8tb/database/CIC-IDS-2017/MachineLearningCSV`

Days:
- Monday-WorkingHours
- Tuesday-WorkingHours
- Wednesday-WorkingHours
- Thursday-WorkingHours-Morning-WebAttacks
- Thursday-WorkingHours-Afternoon-Infilteration
- Friday-WorkingHours-Morning
- Friday-WorkingHours-Afternoon-PortScan
- Friday-WorkingHours-Afternoon-DDos

### 6.3 CIC-IDS-2018

Root path: `/mnt/hdd8tb/database/CIC-IDS-2018`

Days: 02-14-2018 through 02-23-2018 (7 days).

### 6.4 ACI-IoT-2023

Root path: `/mnt/hdd8tb/database/ACI-IoT-2023`

---

## 7. Running Experiments

### 7.1 Full Paper Reproduction (All KNAD Attacks, All Variants)

```bash
# PyTorch backend (default)
python run_experiments.py

# TensorFlow backend
python run_experiments.py --backend tf
```

This runs 12 variants x 6 attacks = 72 experiments with the default configuration
(DBSCAN clustering, Statistical output layer, threshold detector).

### 7.2 Experiment Variants

The 12 model variants (Paper Sec V-C):

| Variant Name | AE Type | Paradigm |
|---|---|---|
| `elm` | ELM | Packet-by-Packet |
| `stat` | Statistical | Packet-by-Packet |
| `conv1d` | Conv1D | TSR |
| `conv2d` | Conv2D | TSR |
| `transformer` | Transformer | TSR |
| `deep_mlp` | Deep MLP | TSR |
| `lstm` | LSTM | TSR |
| `conv1d_ar` | Conv1D | AR |
| `conv2d_ar` | Conv2D | AR |
| `transformer_ar` | Transformer | AR |
| `deep_mlp_ar` | Deep MLP | AR |
| `lstm_ar` | LSTM | AR |

### 7.3 Selective Execution

```bash
# Single attack
python run_experiments.py --attacks Mirai_Botnet

# Specific variants
python run_experiments.py --variants lstm lstm_ar conv1d conv1d_ar

# With specific detector
python run_experiments.py --detector centroid
python run_experiments.py --detector distribution

# Clustering and output layer options
python run_experiments.py --clustering dbscan --output-ae stat
python run_experiments.py --clustering corr --output-ae elm
python run_experiments.py --clustering kmeans

# Other datasets
python run_experiments.py --dataset cic2017
python run_experiments.py --dataset cic2017 --day Monday-WorkingHours
python run_experiments.py --dataset cic2018
python run_experiments.py --dataset aci-iot

# Force recomputation (ignore cached scores)
python run_experiments.py --no-cache
```

### 7.4 CLI Reference

```
--attacks       Attack/day names (default: all 6 paper attacks)
--variants      Variant names (default: all 12)
--clustering    corr | dbscan | kmeans (default: dbscan)
--output-ae     elm | stat (default: stat)
--detector      threshold | centroid | distribution (default: threshold)
--dataset       knad | cic2017 | cic2018 | aci-iot (default: knad)
--day           Specific day for CIC datasets
--backend       torch | tf (default: torch)
--no-cache      Recompute all scores
```

### 7.5 Output Files

- `results/results_<dataset>.json` -- Aggregated metrics per attack per variant.
- `results/<dataset>/<attack>/<variant>_<clustering>_<output_ae>_scores.npy` -- Raw anomaly scores (cached for resume).
- `results/<dataset>/<attack>/roc.png` -- ROC curves per attack.
- `results/experiments.log` -- Full experiment log.

---

## 8. Existing Results Summary

Results from `results/results.json` (KNAD dataset, PyTorch backend, threshold detector):

### 8.1 Best F1 Scores by Attack and Model

| Attack | ELM | Conv1D | Conv2D | Transformer | Deep MLP |
|---|---|---|---|---|---|
| SSDP_Flood | 0.9938 | 0.9943 | 0.9944 | 0.9943 | 0.9944 |
| Mirai_Botnet | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ARP_MitM | 0.7414 | 0.8580 | 0.7541 | 0.8330 | 0.7476 |
| OS_Scan | 0.6579 | 0.6053 | 0.6052 | 0.6153 | 0.6052 |
| Active_Wiretap | 0.6165 | 0.6199 | 0.6186 | 0.6168 | 0.6190 |
| SSL_Renegotiation | 0.5758 | 0.2859 | 0.2859 | 0.2859 | 0.2859 |
| Fuzzing | 0.3741 | 0.4126 | 0.3961 | 0.4071 | 0.4026 |
| Video_Injection | 0.1642 | 0.0869 | 0.0873 | 0.0902 | 0.0873 |
| SYN_DoS | 0.1153 | 0.0408 | 0.0833 | 0.0877 | 0.0696 |

### 8.2 Key Observations

- **SSDP Flood** and **Mirai Botnet** are near-perfectly detected by all models (F1 > 0.99).
- **ARP MitM** shows the largest differentiation: Conv1D (0.858) significantly outperforms ELM (0.741).
- **SSL Renegotiation**, **Video Injection**, and **SYN DoS** are challenging for all models, with windowed DL models sometimes performing worse than packet-by-packet ELM.
- The `elm_reg` (z-score regularized ELM) performs nearly identically to standard `elm` across all attacks.

---

## 9. Verifying Backend Equivalence

### 9.1 Key Architectural Differences Between Backends

The PyTorch and TensorFlow implementations are **not architecturally identical** for most
models. Understanding the differences is critical:

| Aspect | PyTorch | TensorFlow |
|---|---|---|
| **Reconstruction target** | Full window (Conv1D, Conv2D, Transformer, Deep MLP) | Last/next frame only |
| **RMSE computation** | Over `(seq_len, n_visible)` dims | Over `n_visible` dim only |
| **Conv1D encoder** | Single Conv1d + 2 Linear | 2x Conv1D + MaxPool + GAP + Dense |
| **Conv2D encoder** | 2x Conv2d(8,16) + AdaptiveAvgPool | 2x Conv2D(32,n_hidden) + MaxPool + GAP |
| **Transformer** | Symmetric MHA-Dense-MHA (no residual) | MHA + residual + BN + GAP |
| **Deep MLP decoder** | No output activation | Sigmoid output |
| **LSTM** | Architecturally equivalent | Architecturally equivalent |
| **ELM** | Shared NumPy code | Shared NumPy code |
| **Statistical** | Shared NumPy code | Shared NumPy code |

### 9.2 Running Equivalence Tests

```bash
# Run the same attack with both backends
python run_experiments.py --attacks SSDP_Flood --variants elm stat --backend torch
python run_experiments.py --attacks SSDP_Flood --variants elm stat --backend tf

# Compare results
python -c "
import json
with open('results/results_knad.json') as f:
    r = json.load(f)
for attack, variants in r.items():
    for variant, metrics in variants.items():
        print(f'{attack}/{variant}: F1={metrics[\"best_f1\"]:.4f}')
"
```

### 9.3 Shared Components (Framework-Agnostic)

The following produce identical results regardless of backend:

- **ELM autoencoder** (`src/common/autoencoders/elm.py`) -- pure NumPy
- **Statistical detector** (`src/common/autoencoders/statistical_ae.py`) -- pure NumPy
- **Centroid detector** (`src/common/detectors/centroid.py`) -- scikit-learn KMeans
- **Distribution detector** (`src/common/detectors/distribution.py`) -- pure NumPy
- **Feature mapping / clustering** -- shared code
- **Normalization** -- identical min-max formula with same epsilon
- **Windowing** -- `create_windows()` in `src/common/utils.py` using stride tricks
- **Threshold sweep and metrics** -- shared evaluation code

### 9.4 Expected Divergence Sources

For the deep learning models (Conv1D, Conv2D, Transformer, Deep MLP, LSTM), results will
differ between backends due to:

1. **Architectural differences** (see table above)
2. **Weight initialization** -- Different default init schemes (PyTorch vs Keras)
3. **Floating-point ordering** -- GPU non-determinism even with identical architectures
4. **Optimizer implementation details** -- Subtle differences in Adam between frameworks

The LSTM model is the closest to equivalent between backends. ELM and Statistical are
exactly equivalent (shared NumPy code).

---

## 10. Data Processing Pipeline

For each experiment, the data flow is:

```
Raw Packets (AfterImage 115 features)
  |
  v
[1] Feature Mapping Phase (FM_GRACE_PERIOD = 50k packets)
    -> DBSCAN clustering to group correlated features
    -> Creates ensemble structure: 4 autoencoders + 1 output layer
  |
  v
[2] Autoencoder Training Phase (AD_GRACE_PERIOD = 150k packets)
    -> Train each ensemble AE on its feature group
    -> Output layer (ELM or Statistical) trained on ensemble RMSE scores
  |
  v
[3] Execution Phase (EXECUTION_WINDOW = 400k packets)
    -> Compute per-packet anomaly scores
    -> Apply detector (threshold / centroid / distribution)
    -> Evaluate: F1, Recall, FPR, WindowDiff
```

### Windowing Details

For all deep learning models (Conv1D, Conv2D, Transformer, Deep MLP, LSTM):

- Sliding windows of size `seq_len` are created over the normalized data.
- A `back_window` of size `seq_len - 1` is saved after training to maintain continuity during execution.
- Training shuffles the DataLoader; evaluation never shuffles (Bug Fix #2).

### TSR vs AR Paradigm

- **TSR (Time-Series Reconstruction):** Window = `x[i:i+seq_len]`, target = `x[i+seq_len-1]` (last frame) or full window.
- **AR (Autoregressive):** Window = `x[i:i+seq_len]`, target = `x[i+seq_len]` (next frame, one step ahead).

---

## 11. Troubleshooting

### Out of Memory
Reduce `BATCH_SIZE` in `src/common/config.py` or use `--variants` to run fewer models at a time.

### Missing Dataset
Update `DATA_ROOT`, `CIC2017_ROOT`, `CIC2018_ROOT`, or `ACI_IOT_ROOT` in `src/common/config.py` to point to your local dataset paths.

### Resuming Interrupted Runs
Scores are cached as `.npy` files. Re-running the same command will skip already-computed variants. Use `--no-cache` to force recomputation.

### Reproducing Exact Numbers
Due to GPU non-determinism, exact floating-point reproduction requires:
1. Using CPU only (`DEVICE = "cpu"` in config.py)
2. Same library versions (PyTorch, TF, NumPy, scikit-learn)
3. Same random seed (1234, already set)
4. Single-threaded execution for NumPy: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_experiments.py`
