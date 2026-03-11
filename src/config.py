"""Central configuration for the kitsune-zd experiments."""
import os

DATA_ROOT = "/mnt/hdd8tb/database/KITSUNE"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ATTACKS = [
    "Active_Wiretap",
    "ARP_MitM",
    "Fuzzing",
    "Mirai_Botnet",
    "OS_Scan",
    "SSDP_Flood",
    "SSL_Renegotiation",
    "SYN_DoS",
    "Video_Injection",
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

# Hyperparameters (match original NDSS'18 paper defaults)
FM_GRACE_PERIOD = 100_000
AD_GRACE_PERIOD = 100_000
MAX_AE_SIZE = 10
HIDDEN_RATIO = 0.75
LEARNING_RATE_ELM = 0.1
LEARNING_RATE_DL = 0.001
SEQUENCE_LENGTH = 500
EXECUTION_WINDOW = 10_000
BATCH_SIZE = 32
DEVICE = "cuda"
N_FEATURES = 115
SEED = 1234

EPSILON = 1e-16
