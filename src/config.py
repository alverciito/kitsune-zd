"""Central configuration for the kitsune-zd experiments."""
import os

DATA_ROOT = "/mnt/hdd8tb/database/KITSUNE"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# KNAD attacks used in the IEEE paper (Table I, Fig. 7)
ATTACKS = [
    "SSDP_Flood",
    "Active_Wiretap",
    "Fuzzing",
    "OS_Scan",
    "Video_Injection",
    "SSL_Renegotiation",
]

# Maps attack name -> subdirectory name in DATA_ROOT
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

# Hyperparameters (match IEEE paper Table II)
FM_GRACE_PERIOD = 50_000       # Clustering Train Phase (50 kPacket)
AD_GRACE_PERIOD = 150_000      # AE Train Phase (150 kPacket)
MAX_AE_SIZE = 4                # Number of ensemble autoencoders (4+1)
HIDDEN_RATIO = 0.22            # 78% compression ratio (keep 22%)
LEARNING_RATE = 0.001          # ADAM lr for all models
SEQUENCE_LENGTH = 800          # Context window (packets)
EXECUTION_WINDOW = 400_000     # Execution batch size (400 kPacket)
BATCH_SIZE = 32
DEFAULT_CLUSTERING = 'dbscan'  # Paper uses DBSCAN for feature mapping
DEFAULT_OUTPUT_AE = 'stat'     # Paper uses Statistical output layer
DEVICE = "cuda"
N_FEATURES = 115
SEED = 1234

EPSILON = 1e-16

# === Multi-dataset paths ===
CIC2017_ROOT = "/mnt/hdd8tb/database/CIC-IDS-2017/MachineLearningCSV"
CIC2018_ROOT = "/mnt/hdd8tb/database/CIC-IDS-2018"
ACI_IOT_ROOT = "/mnt/hdd8tb/database/ACI-IoT-2023"

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

CIC2018_DAYS = [
    "02-14-2018",
    "02-15-2018",
    "02-16-2018",
    "02-20-2018",
    "02-21-2018",
    "02-22-2018",
    "02-23-2018",
]
