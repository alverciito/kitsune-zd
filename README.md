# kitsune-zd

Reproduction and extension of **KitNET** ([Mirsky et al., NDSS 2018](https://www.ndss-symposium.org/ndss-paper/kitsune-an-ensemble-of-autoencoders-for-online-network-intrusion-detection/)) for network intrusion detection. Compares **7 autoencoder architectures**, **2 paradigms** (TSR / AR), **3 clustering methods**, **2 output layers**, and **3 detector types** across **4 datasets** — supporting 216+ experiment configurations.

Ported from an internal TensorFlow 2.x codebase; fixes 4 critical bugs and extends the pipeline with new models, clustering algorithms, and post-hoc detectors described in the companion IEEE paper.

---

## Architecture

KitNET is an **ensemble of autoencoders** with three phases:

```
Phase 1 — Feature Mapping  (100K samples)
  Cluster N features into K groups via CorClust / DBSCAN / KMeans.

Phase 2 — AD Training  (100K samples)
  Train K ensemble autoencoders (one per cluster)
  + 1 output layer (ELM or Statistical) that aggregates their RMSE scores.

Phase 3 — Execution  (remaining samples)
  packet → ensemble AEs → RMSE vector → output layer → anomaly score
  Optional: CentroidDetector or DistributionDetector post-processing.
```

---

## Models

### Online (packet-by-packet, NumPy)

| Variant    | Architecture                                  | Notes                         |
|------------|-----------------------------------------------|-------------------------------|
| `elm`      | `sigmoid(Wx+b) → sigmoid(W'h+b')`            | Original Kitsune (NDSS '18).  |
| `elm_reg`  | Same, z-score regularized input               | Tests regularization effect.  |
| `stat`     | `score = Σ\|μ − x\| / (σ + ε)`               | Statistical baseline (Eq. 3). |

### Windowed (batch, seq_len = 500)

| Variant       | Architecture                                               | Framework  |
|---------------|------------------------------------------------------------|------------|
| `conv1d`      | Conv1D(n_vis→3, k=500, same) → Dense → Dense              | PyTorch    |
| `conv2d`      | Conv2d(1→8→16, 3×3) → AdaptivePool → Dense → Dense        | PyTorch    |
| `transformer` | MHA → Dense(σ) → Dense(relu) → MHA                        | PyTorch    |
| `deep_mlp`    | Flatten → Linear(//4) → Linear(h) → Linear(//4) → Reshape | PyTorch    |
| `lstm`        | LSTM(n_h) → Dense(relu) → Dense(sigmoid)                  | TensorFlow (see `tf/`) |

All windowed variants support both **TSR** (reconstruct the window) and **AR** (predict next frame). Append `_ar` to the variant name for autoregressive mode (e.g. `lstm_ar`).

---

## Clustering

| Flag      | Algorithm                               |
|-----------|-----------------------------------------|
| `corr`    | Incremental correlation + hierarchical  |
| `dbscan`  | DBSCAN with iterative ε tuning         |
| `kmeans`  | KMeans on transposed feature matrix     |

## Output layers

| Flag   | Description                         |
|--------|-------------------------------------|
| `elm`  | ELM autoencoder aggregation         |
| `stat` | Statistical: `\|μ − x\| / (σ + ε)` |

## Detectors (post-processing)

| Flag           | Description                                                   |
|----------------|---------------------------------------------------------------|
| `threshold`    | Sweep thresholds on log-transformed scores (default)          |
| `centroid`     | K-means++ on score distribution → dist-to-centroid + mean filter |
| `distribution` | Eq. 4 harmonic-mean of (μ, σ) deviation + median filter      |

---

## Datasets

| Dataset       | Features | Scenarios | Label source         |
|---------------|:--------:|:---------:|----------------------|
| KITSUNE/KNAD  |   115    | 9 attacks | AfterImage statistics|
| CIC-IDS-2017  |    78    | 8 days    | CICFlowMeter         |
| CIC-IDS-2018  |    79    | 7 days    | CICFlowMeter         |
| ACI-IoT-2023  | variable | 1         | IoT traffic          |

---

## Metrics

- **F1 / Recall / FPR** — threshold sweep on log-scores
- **WindowDiff** — segmentation metric (Sec. V-B of IEEE paper)

---

## Bug fixes (vs. original TF codebase)

| #  | Severity | Description |
|----|----------|-------------|
| 1  | Critical | `elm_reg` return value was never captured — results were stale copies of the previous variant. |
| 2  | High     | Evaluation DataLoaders used `shuffle=True`, misaligning RMSE scores with ground-truth labels. |
| 3  | Medium   | Output layer received inconsistent RMSE matrix shapes across variants. |
| 4  | Medium   | ELM decoder weights (`W_prime`) diverged from encoder — never updated after SGD steps. |

Additional fixes:
- **Optimizer**: SGD → Adam (lr=0.001, β₁=0.9, β₂=0.999) in all DL models.
- **Precision/Recall guards**: `tp+fp=0` and `tp+fn=0` no longer produce inflated metrics via ε-smoothing.
- **Seed propagation**: `torch.manual_seed(1234)` / `tf.random.set_seed(1234)` in every model for reproducibility.
- **CorClust single-feature guard**: No longer crashes with `n_features=1`.
- **KMeans cluster cap**: `n_clusters = min(requested, n_features)` prevents sklearn ValueError.

---

## Project structure

```
kitsune-zd/
├── run_experiments.py             # Main CLI entry point
├── conftest.py                    # pytest path config
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py                # Public API re-exports
│   ├── config.py                  # Hyperparameters, dataset paths
│   ├── database.py                # Loaders: KNAD, CIC-2017/2018, ACI-IoT
│   ├── kitnet.py                  # KitNET 3-phase pipeline orchestrator
│   ├── detector.py                # threshold_sweep, windowdiff, plot_roc
│   ├── utils.py                   # sigmoid, windowing (TSR + AR), RMSE
│   ├── autoencoders/
│   │   ├── __init__.py
│   │   ├── elm.py                 # ELM online SGD          (NumPy)
│   │   ├── statistical_ae.py      # Statistical Eq. 3       (NumPy)
│   │   ├── conv1d_ae.py           # Conv1D                  (PyTorch)
│   │   ├── conv2d_ae.py           # Conv2D                  (PyTorch)
│   │   ├── transformer_ae.py      # Transformer MHA         (PyTorch)
│   │   ├── deep_mlp_ae.py         # Deep MLP                (PyTorch)
│   │   └── (lstm moved to tf/)     # See tf/ for TensorFlow models
│   ├── clustering/
│   │   ├── __init__.py            # get_clustering() factory
│   │   ├── corclust.py            # Incremental correlation
│   │   ├── dbscan_clust.py        # DBSCAN + eps tuning
│   │   └── kmeans_clust.py        # KMeans wrapper
│   └── detectors/
│       ├── __init__.py
│       ├── centroid.py            # CentroidDetector (K-means++)
│       ├── distribution.py        # DistributionDetector (Eq. 4)
│       └── filters.py             # mean_filter, median_filter
├── tf/                               # TensorFlow models (optional dependency)
│   ├── __init__.py
│   ├── lstm_ae.py                 # LSTM encoder-decoder    (TensorFlow/Keras)
│   └── requirements.txt           # pip install -r tf/requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_utils.py              # Windowing, normalization, sigmoid
│   ├── test_autoencoders.py       # All 6 AE variants (77 tests)
│   ├── test_clustering.py         # CorClust, DBSCAN, KMeans (19 tests)
│   ├── test_detectors.py          # Detectors, filters, WindowDiff (25 tests)
│   └── test_kitnet.py             # End-to-end pipeline (17 tests)
└── results/                       # Generated scores, ROC plots, JSON metrics
```

---

## Usage

```bash
pip install -r requirements.txt

# Default: 6 KNAD attacks × 12 variants (paper configuration)
python run_experiments.py

# Specific attacks and variants
python run_experiments.py --attacks Video_Injection SSL_Renegotiation
python run_experiments.py --variants elm conv1d lstm lstm_ar

# Override clustering and output layer (defaults are DBSCAN + Statistical)
python run_experiments.py --clustering corr --output-ae elm

# Post-hoc detector
python run_experiments.py --detector centroid

# Other datasets
python run_experiments.py --dataset cic2017 --day Monday-WorkingHours
python run_experiments.py --dataset cic2018
python run_experiments.py --dataset aci-iot

# Force recompute
python run_experiments.py --no-cache
```

## Tests

```bash
python -m pytest tests/ -v
```

170 tests covering all autoencoders, clustering algorithms, detectors, filters, windowing, metrics, and full end-to-end pipeline — including anomaly sensitivity checks (anomaly data must score higher than benign), edge cases (1 feature, constant input, extreme values), and reproducibility.

---

## Hyperparameters (IEEE paper Table II)

| Parameter              | Symbol        | Value        | Source            |
|------------------------|---------------|-------------:|-------------------|
| Clustering Train Phase | G_FM          |       50,000 | IEEE paper Tab II |
| AE Train Phase         | G_AD          |      150,000 | IEEE paper Tab II |
| Total Train Phase      | G_FM + G_AD   |      200,000 | IEEE paper Tab II |
| Number of Autoencoders | N + 1         |          4+1 | IEEE paper Tab II |
| Compression Ratio      | 1 - r_h       |          78% | IEEE paper Tab II |
| Hidden ratio           | r_h           |         0.22 | 1 - 0.78         |
| Sequence length        | L_seq         |          800 | IEEE paper Tab II |
| Clustering algorithm   | —             |       DBSCAN | IEEE paper Tab II |
| Output AE type         | —             |  Statistical | IEEE paper Tab II |
| Normalization          | —             |      Min-max | IEEE paper Tab II |
| Optimizer              | —             |      GD-ADAM | IEEE paper Tab II |
| Learning rate          | α             |        0.001 | IEEE paper Tab II |
| Adam betas             | (β₁, β₂)     | (0.9, 0.999) | IEEE paper Tab II |
| LR scheduler           | —             |           No | IEEE paper Tab II |
| Weight decay           | —             |           No | IEEE paper Tab II |
| Reconstruction loss    | —             |          MSE | IEEE paper Tab II |
| Batch size             | B             |           32 | Implementation    |
| Execution window       | W_exec        |      400,000 | Original TF repo |
| Training epochs        | E             |            1 | Single pass       |
| Epsilon                | ε             |        1e-16 | Numerical safety  |
| Random seed            | —             |         1234 | Reproducibility   |
| Centroid clusters      | K             |            8 | IEEE paper IV-D   |
| Detection window       | W             |       10,000 | IEEE paper IV-D   |
| Filter window          | —             |          100 | Empirical         |

---

## Origin

Extended from an internal research project at **Universidad de Alcala, Escuela Politecnica Superior**. Original codebase: TensorFlow 2.x (backup at `/mnt/hdd8tb/backup/acid137/PythonProjects/anomaly/`). This version ports DL models to PyTorch 2.x (LSTM remains TF-native), adds DBSCAN/KMeans clustering, CentroidDetector/DistributionDetector, multi-dataset support, WindowDiff metric, AR paradigm, and fixes all documented bugs.
