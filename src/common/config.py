"""
KitNET configuration constants.

Hyperparameters and dataset paths for the kitsune-zd experiments, based on:
"Improving Zero-Day Network Intrusion Detection with Context-Aware Autoencoders"
IEEE Transactions on Information Forensics and Security (TIFS), 2025.

Constants in this module correspond to Table II of the paper unless otherwise noted.
Attack names and directories correspond to the KITSUNE dataset (Table I, Fig. 7).
Multi-dataset paths support evaluation on CIC-IDS-2017, CIC-IDS-2018, and ACI-IoT-2023.
"""
import os

# Root directory containing KITSUNE dataset attack subdirectories.
DATA_ROOT = "/mnt/hdd8tb/database/KITSUNE"

# Output directory for experiment results (JSON metrics, ROC plots).
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# KNAD attacks used in the IEEE paper (Table I, Fig. 7).
# These six attacks are the primary evaluation targets.
ATTACKS = [
    "SSDP_Flood",
    "Active_Wiretap",
    "Fuzzing",
    "OS_Scan",
    "Video_Injection",
    "SSL_Renegotiation",
]

# Maps canonical attack name -> filesystem subdirectory name in DATA_ROOT.
# Includes all nine KITSUNE attacks; ATTACKS above selects the subset used.
ATTACK_DIRS = {
    "Active_Wiretap": "Active Wiretap",
    "ARP_MitM": "ARP MitM",
    "Fuzzing": "Fuzzing",
    "Mirai_Botnet": "Mirai Botnet",
    "OS_Scan": "OS Scan",
    "SSDP_Flood": "SSDP Flood",
    "SSL_Renegotiation": "SSL Renegotiation",
    "SYN_DoS": "SYN DoS",
    "Video_Injection": "Video Injection",
}

# ---- Hyperparameters (Table II of the IEEE paper) ----

FM_GRACE_PERIOD = 50_000       # Table II: Clustering/Feature-Mapping train phase (50 kPacket)
AD_GRACE_PERIOD = 150_000      # Table II: Autoencoder train phase (150 kPacket)
MAX_AE_SIZE = 4                # Table II: Max features per ensemble AE (4+1 architecture)
HIDDEN_RATIO = 0.22            # Table II: Hidden-layer ratio -> 78% compression (keep 22%)
LEARNING_RATE = 0.001          # Table II: ADAM learning rate for all models
SEQUENCE_LENGTH = 800          # Table II: Context window size in packets for TSR/AR modes
EXECUTION_WINDOW = 400_000     # Table II: Execution batch size (400 kPacket)
BATCH_SIZE = 32                # Table II: Mini-batch size for deep AE training
DEFAULT_CLUSTERING = 'dbscan'  # Table II: Default feature-mapping algorithm (DBSCAN)
DEFAULT_OUTPUT_AE = 'stat'     # Table II: Default output layer type (Statistical)
DEVICE = "cuda"                # PyTorch device for deep AE variants (Conv1D, Transformer)
N_FEATURES = 115               # Number of AfterImage features extracted per packet
SEED = 1234                    # Global random seed for reproducibility

# Small constant to prevent division by zero in normalization and scoring.
EPSILON = 1e-16

# ---- Multi-dataset paths for cross-dataset evaluation ----

# CIC-IDS-2017: flow-based dataset with 78 features per flow.
CIC2017_ROOT = "/mnt/hdd8tb/database/CIC-IDS-2017/MachineLearningCSV"

# CIC-IDS-2018: updated version of CIC-IDS-2017 with additional attacks.
CIC2018_ROOT = "/mnt/hdd8tb/database/CIC-IDS-2018"

# ACI-IoT-2023: IoT-specific intrusion detection dataset.
ACI_IOT_ROOT = "/mnt/hdd8tb/database/ACI-IoT-2023"

# CIC-IDS-2017 day labels corresponding to individual capture files.
CIC2017_DAYS = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-WorkingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos",
]

# CIC-IDS-2018 day labels corresponding to individual capture files.
CIC2018_DAYS = [
    "02-14-2018",
    "02-15-2018",
    "02-16-2018",
    "02-20-2018",
    "02-21-2018",
    "02-22-2018",
    "02-23-2018",
]
