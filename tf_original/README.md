# TensorFlow Original Codebase

Código original TensorFlow 2.x del paper IEEE TIFS (July 2025):

**"Improving Zero-Day Network Intrusion Detection with Context-Aware Autoencoders"**

Este código fue el que generó **todos los resultados** de la Fig. 7 del paper
(216 experimentos). El codebase `kitsune-zd` (raíz del repo) es la reescritura
PyTorch de este código.

## Origen

Copiado de: `/mnt/hdd8tb/backup/acid137/PythonProjects/anomaly/`

Resultados originales (no incluidos, ~24 GB de .npy):
`/mnt/hdd8tb/backup/acid137/PythonProjects/anomaly/experiments/frames/*/results/`

## Estructura

```
tf_original/
├── src/
│   ├── database/              # Loaders de datasets (KNAD, CIC, ACI-IoT)
│   ├── detectors/             # Detectores post-hoc (vacío, se usan en show_stats)
│   ├── models/
│   │   ├── kitnet.py          # KitNET core (3 fases)
│   │   ├── oopsie.py          # NETWORK_TYPES registry + validación
│   │   ├── __special__.py     # Logging y constantes
│   │   ├── cluster/           # CorClust, DBSCAN, KMeans, Random
│   │   ├── networks/          # Autoencoders TF:
│   │   │   ├── conv1d.py      #   Conv1DAutoencoder (AR/TSR)
│   │   │   ├── conv2d.py      #   Conv2DAutoencoder (AR/TSR)
│   │   │   ├── lstm.py        #   LSTMAutoencoder (AR/TSR)
│   │   │   ├── mha.py         #   MHAAutoencoder (AR/TSR)
│   │   │   ├── multilayer.py  #   MLPAutoencoder (AR/TSR)
│   │   │   ├── original_kitsune.py  # ThreeLayerMLP (KitNET original, PbP)
│   │   │   └── stdev.py       #   StatisticalAnomaly (PbP)
│   │   └── utils/
│   │       ├── sequential.py  # create_windowed_data / create_windowed_data_ar
│   │       ├── data.py        # Utilidades de datos
│   │       └── gather_layer.py
│   └── old_models/            # Versiones anteriores (conv1d, mha, kitnet adaptados)
├── experiments/
│   └── frames/
│       ├── launch_all.py      # Launcher principal (CONFIG + model_experiment_launcher)
│       ├── kitsune/           # Experimentos KNAD
│       ├── CIC2017/           # Experimentos CIC-IDS-2017
│       ├── CIC2018/           # Experimentos CIC-IDS-2018
│       ├── CIC2019/           # Experimentos CIC-IDS-2019
│       └── ACI-IOT-2023/      # Experimentos ACI-IoT-2023
├── test/
│   └── models/
│       ├── test_kitnet.py
│       ├── test_models.py
│       └── test_preprocessing.py
└── backup/
    └── dev/                   # Variantes experimentales de autoencoders
```

## Modelos disponibles (NETWORK_TYPES)

```python
NETWORK_TYPES = {
    'mha': MHAAutoencoder,        # Multi-Head Attention
    'original': ThreeLayerMLP,    # KitNET original (ELM-like)
    'lstm': LSTMAutoencoder,      # LSTM
    'mlp': MLPAutoencoder,        # Deep MLP
    'conv1d': Conv1DAutoencoder,  # Conv1D
    'conv2d': Conv2DAutoencoder,  # Conv2D
    'stat': StatisticalAnomaly    # Statistical (Eq. 3)
}
```

## Configuración del paper (launch_all.py)

```python
CONFIG = {
    'train_period': 150,         # kPacket (AE training)
    'clustering_period': 50,     # kPacket (feature mapping)
    'sequence_length': 800,      # Packet context window
    'hidden_ratio': 0.22,        # 78% compression
    'autoencoder_size': 4,       # 4+1 ensemble
    'clustering': 'dbscan',
    'output_ae_type': 'stat',
    'execution_window': 400,     # kPacket
}
```

## Ejecución original

Cada experimento se ejecutó con GPU:1 (TensorFlow):
```python
with tf.device('/GPU:1'):
    for is_ar in [True, False]:          # 2 paradigmas
        for model_type in NETWORK_TYPES:  # 7 modelos
            kn = KitNET(...)
            for packet in x:
                kn.process(packet)
            kn.show_stats(y, save=path)
```

Resultados: 14 score files por dataset (`{modelo}_ar_{True|False}.npy`)
