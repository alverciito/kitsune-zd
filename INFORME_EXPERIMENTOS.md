# Informe de Experimentos: KitNET (kitsune-zd) vs Paper IEEE

## Paper

**"Improving Zero-Day Network Intrusion Detection with Context-Aware Autoencoders"**
- IEEE Transactions on Information Forensics and Security, Vol. , No. , July 2025
- Autores: Alberto Palomo-Alonso, Khalid Achbab-Gueriguer, Antonio Portilla-Figueras,
  Silvia Jiménez-Fernández, Enrique Alexandre-Cortizo, Sancho Salcedo-Sanz
- Desarrollado en: Isdefe-UAH Observatory in ICT and Artificial Intelligence

## Resumen del paper

El paper propone una versión mejorada de KitNET que incorpora autoencoders
context-aware (Conv1D, Conv2D, MHA, LSTM, MLP) para detección de intrusiones
zero-day. A diferencia del KitNET original (packet-by-packet), estos modelos
procesan secuencias temporales de paquetes en dos paradigmas:

- **TSR** (Time Series Reconstruction): reconstruir la secuencia completa.
- **AR** (Autoregressive): predecir el siguiente paquete a partir del historial.
- **PbP** (Packet-by-Packet): KitNET original (ELM) y Statistical.

Se evalúan con dos detectores post-hoc:
- **KMD**: Centroid Detector + Mean filter (K-means++ sobre Detection Frame).
- **MVD**: Distribution Detector + Median filter (Eq. 4, media armónica).

---

## Arquitectura del paper (Sec. IV)

### Modelos (12 por dataset)

| # | Paradigma | Modelo       | Descripción                              |
|---|-----------|-------------|------------------------------------------|
| 1 | AR        | MLP         | Deep MLP autoregresivo                   |
| 2 | TSR       | MLP         | Deep MLP reconstrucción completa         |
| 3 | AR        | Conv1D      | Conv1D autoregresivo                     |
| 4 | TSR       | Conv1D      | Conv1D reconstrucción completa           |
| 5 | AR        | Conv2D      | Conv2D autoregresivo                     |
| 6 | TSR       | Conv2D      | Conv2D reconstrucción completa           |
| 7 | AR        | MHA         | Multi-Head Attention autoregresivo       |
| 8 | TSR       | MHA         | Multi-Head Attention reconstrucción      |
| 9 | AR        | LSTM        | LSTM autoregresivo                       |
|10 | TSR       | LSTM        | LSTM reconstrucción completa             |
|11 | PbP       | KitNET      | ELM original (sin contexto temporal)     |
|12 | PbP       | STAT        | Statistical (Eq. 3: |mu-x|/(sigma+eps)) |

### Datasets (9 totales, Table I)

| Dataset                | Filas      | Cols | % Malicioso |
|------------------------|------------|------|-------------|
| KNAD - SSDP Flood      | 4 077 265  | 115  | 35.31%      |
| KNAD - Active Wiretap   | 2 278 688  | 115  | 40.52%      |
| KNAD - Fuzzing          | 2 224 138  | 115  | 19.29%      |
| KNAD - OS Scan          | 1 697 850  | 115  | 03.87%      |
| KNAD - Video Injection  | 2 472 400  | 115  | 04.15%      |
| KNAD - SSL Renegotiation| 2 207 570  | 115  | 04.20%      |
| ACI-IoT-2023           | 1 231 411  | 84   | 73.26%      |
| CIC-2017 (2 días)      | ~500k-700k | 78   | variable    |
| CIC-2018 (2 días)      | ~1M        | 79   | variable    |

### Hiperparámetros (Table II)

| Parámetro               | Valor       |
|--------------------------|-------------|
| Clustering Train Phase   | 50 000 pkts |
| AE Train Phase           | 150 000 pkts|
| Total Train Phase        | 200 000 pkts|
| Num. Autoencoders        | 4 + 1       |
| Compression Ratio        | 78%         |
| Sequence Length           | 800 pkts    |
| Clustering Algorithm     | DBSCAN      |
| Output Autoencoder Type  | Statistical |
| Learning Rate            | 0.001       |
| Optimizer                | GD-ADAM     |
| Reconstruction Loss      | MSE         |

### Total de experimentos: 216

9 datasets × 12 modelos × 2 detectores = 216

---

## Estado de la implementación (kitsune-zd, PyTorch)

### Código: COMPLETO

| Componente               | Archivo                          | Estado |
|--------------------------|----------------------------------|--------|
| KitNET core              | `src/kitnet.py`                  | OK     |
| Conv1D AE                | `src/autoencoders/conv1d_ae.py`  | OK     |
| Conv2D AE                | `src/autoencoders/conv2d_ae.py`  | OK     |
| Transformer (MHA)        | `src/autoencoders/transformer_ae.py` | OK |
| Deep MLP                 | `src/autoencoders/deep_mlp_ae.py`| OK     |
| ELM (KitNET original)    | `src/autoencoders/elm.py`        | OK     |
| Statistical (STAT)       | `src/autoencoders/statistical_ae.py` | OK |
| LSTM                     | `tf/lstm_ae.py`                  | OK (TensorFlow) |
| CentroidDetector (KMD)   | `src/detectors/centroid.py`      | OK     |
| DistributionDetector (MVD)| `src/detectors/distribution.py` | OK     |
| Filtros mean/median      | `src/detectors/filters.py`       | OK     |
| Database loaders         | `src/database.py`                | OK (KNAD, CIC2017, CIC2018, ACI-IoT) |
| Experiment runner        | `run_experiments.py`             | OK (12 variantes, 3 detectores, 4 datasets) |

`run_experiments.py` define las 12 variantes exactas del paper:
```python
DEFAULT_VARIANTS = [
    'deep_mlp_ar', 'deep_mlp', 'conv1d_ar', 'conv1d', 'conv2d_ar', 'conv2d',
    'transformer_ar', 'transformer', 'lstm_ar', 'lstm', 'elm', 'stat',
]
```

---

## Experimentos ejecutados vs pendientes

### Ejecutados (54 score files, ~882 MB)

- Fecha: 10-11 Marzo 2026 (~9.5 horas)
- 6 variantes TSR solamente: `elm`, `elm_reg`, `conv1d`, `conv2d`, `transformer`, `deep_mlp`
- 9 ataques KNAD (incluye 3 extras no del paper: Mirai_Botnet, ARP_MitM, SYN_DoS)
- Detector: solo `threshold` (sin post-procesado)
- Resultados en: `results/{attack}/{variant}_scores.npy`

### NO ejecutados (pendientes para reproducir el paper)

| Categoría                  | Experimentos faltantes |
|----------------------------|------------------------|
| Variantes AR (5 modelos × 6 ataques) | 30  |
| LSTM TSR + AR (2 × 6 ataques)        | 12  |
| STAT PbP (1 × 6 ataques)             | 6   |
| CIC-2017 (12 modelos × 2 días)       | 24  |
| CIC-2018 (12 modelos × 2 días)       | 24  |
| ACI-IoT-2023 (12 modelos × 1)        | 12  |
| Detectores KMD/MVD (×2 todo)         | ×2  |
| **Total aproximado**                  | **~216** |

---

## Resultados del backup TensorFlow original

Los resultados completos del paper se generaron con el código TensorFlow en:
```
/mnt/hdd8tb/backup/acid137/PythonProjects/anomaly/experiments/frames/
```

| Dataset       | Directorio       | Tamaño | Variantes por ataque                |
|---------------|------------------|--------|-------------------------------------|
| KNAD (6 atq.) | `kitsune/results/` | 13 GB | 14: 7 modelos × {ar_True, ar_False} |
| CIC-2017      | `CIC2017/results/` | 2.6 GB | 14 score files                      |
| CIC-2018      | `CIC2018/results/` | 5.6 GB | 14 score files                      |
| ACI-IoT-2023  | `ACI-IOT-2023/results/` | 1.4 GB | 14 score files                |
| **Total**     |                  | **~24 GB** |                                |

Nomenclatura backup: `{modelo}_ar_{True|False}.npy`
- Modelos: `conv1d`, `conv2d`, `lstm`, `mha`, `mlp`, `original` (ELM), `stat`

### Otras copias del proyecto en el servidor

| Ubicación | Descripción |
|-----------|-------------|
| `/mnt/hdd8tb/backup/acid137/PythonProjects/anomaly/` | Codebase TF original (24 GB resultados) |
| `/mnt/hdd8tb/backup/acid137/PythonProjects/kitsune-hw/` | Versión hardware/acelerada |
| `/mnt/hdd8tb/backup/acid137/Desktop/kitsune-zd/` | Copia con venv y tests adaptados |
| `/mnt/hdd8tb/backup/acid137/Desktop/npdata/khalid/modelos_paper/` | Versión para el paper |

---

## Cómo reproducir el paper completo

```bash
# 1. KNAD completo (6 ataques × 12 modelos × 2 detectores)
python run_experiments.py --detector centroid
python run_experiments.py --detector distribution

# 2. CIC-2017
python run_experiments.py --dataset cic2017 --detector centroid
python run_experiments.py --dataset cic2017 --detector distribution

# 3. CIC-2018
python run_experiments.py --dataset cic2018 --detector centroid
python run_experiments.py --dataset cic2018 --detector distribution

# 4. ACI-IoT-2023
python run_experiments.py --dataset aci-iot --detector centroid
python run_experiments.py --dataset aci-iot --detector distribution
```

Tiempo estimado: ~48-72 horas (basado en los ~10h del subset actual).

---

## Conclusiones clave del paper (Sec. VI-VII)

1. **MLP** obtiene las mejores métricas globales (F1 ~1.0 en Video Injection, SSDP Flood).
2. **Conv1D/Conv2D** son muy similares; Conv1D ligeramente superior y más ligero.
3. **MHA** es el modelo más resiliente (nunca el peor en ningún dataset).
4. **LSTM** funciona bien en datasets simples pero mal en tráfico caótico (CIC-2018).
5. **KitNET (ELM)** baseline sorprendentemente competitivo a pesar de no usar contexto.
6. **STAT** rápido y efectivo en tráfico caótico, pero limitado en patrones complejos.
7. **AR vs TSR**: los mejores resultados tienden a ser AR, pero no hay superioridad clara.
8. **KMD vs MVD**: depende del caso; KMD mejor para multi-banda, MVD para banda única.

---

*Informe generado el 2026-03-20.*
*Datos del servidor: acid137@srv (GPU, /mnt/hdd8tb/database/).*
