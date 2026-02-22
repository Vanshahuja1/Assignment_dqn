# params.py  (Stage-1: Geometry + SINR attach + Edge mapping + PDU session requests)
# + Stage-3 additions: PDF resource model helpers, Home budgets, DQN configs
#
# ✅ UPDATED FOR FASTER TRAINING (only RL/DQN knobs changed)
# - smaller batch
# - smaller replay buffer
# - learn earlier (smaller warmup)
# - train less frequently
# - target update less often
# - faster epsilon decay
# NOTE: This does NOT change topology, arrivals, resource budgets, or PDF logic.

# -----------------------------
# Simulation area (meters)
# -----------------------------
AREA_WIDTH = 3000
AREA_HEIGHT = 3000

# -----------------------------
# Operators
# -----------------------------
NUM_OPERATORS = 2  # Operator 0 and 1

# -----------------------------
# BS Deployment
# -----------------------------
BS_DENSITY_PER_KM2 = [2.0, 2.0]  # per operator (2 BS/km^2)
BS_TYPE_RATIO = [
    [1, 0, 0, 0],  # Operator 0: Macro only
    [1, 0, 0, 0]   # Operator 1: Macro only
]

BS_TYPES = ["MACRO", "MICRO", "PICO", "FEMTO"]

# BS Transmit Powers by type (Watt)
BS_TX_POWER_WATT = {
    "MACRO": 20,
    "MICRO": 5,
    "PICO": 2,
    "FEMTO": 0.1
}

# Heights (meters) (used by pathloss if needed)
BS_HEIGHT = 50
UE_HEIGHT = 1.5

# Frequency per operator (MHz)
OPERATING_FREQUENCY = [900, 1200]

# Noise power for SINR (dBm)
NOISE_POWER_DBM = -104

# Association rule
# True  : user picks best SINR among own operator BSs only
# False : users consider all BSs (cross-operator association allowed)
ASSOC_OWN_OPERATOR_ONLY = False

# -----------------------------
# Edge servers (UPF compute nodes)
# -----------------------------
TOTAL_EDGE_SERVERS = 10

# Not shared between operators. Choose a split that sums to 10.
EDGE_SERVERS_PER_OPERATOR = [5, 5]  # op0=5, op1=5

# Edge placement: random in same area
EDGE_PLACEMENT = "random"

# Map BS -> nearest edge of SAME operator
BS_TO_EDGE_MAPPING = "nearest_same_operator"

# -----------------------------
# Time / slots
# -----------------------------
DT = 1.0
NUM_SLOTS = 100

# -----------------------------
# PDU session request model (Step 2.1 & Table 1)
# -----------------------------
SERVICE_TYPES = ["URLLC", "eMBB", "mMTC"]
SERVICE_PI = [0.2, 0.3, 0.5]  # URLLC set to 0.2 as per client request

# Table 1 Parameters
SERVICE_RATES_KBPS = {
    "URLLC": 64,
    "eMBB": 500,
    "mMTC": 1024
}

SERVICE_LATENCY_REQ_MS = {
    "URLLC": 20,
    "eMBB": 500,
    "mMTC": 1000
}

# CPU cost per session (mC/session/slot) from Table 1
SERVICE_CPU_COST_MC = {
    "URLLC": 0.154,
    "eMBB": 1.200,
    "mMTC": 2.458
}

# Operator split for users
USER_OPERATOR_RATIO = (5, 5)

# Each slot, each user generates a new session with this probability.
P_SESSION_REQUEST = 0.15  # Tripled to create massive load

# -----------------------------
# Latency: gNB -> edge hop (distance-based transport)
# -----------------------------
# Propagation speed in m/s (Eq. 36)
V_F = 2e8  

# -----------------------------
# Resource budgets (Step 2.4)
# -----------------------------
# Total CPU capacity per edge (mC per slot)
EDGE_CPU_BUDGET_MC = 50  # Lowered significantly to force congestion

# Total Memory capacity per edge (MB)
EDGE_MEM_BUDGET_MB = 2048  

# Memory consumed per session (MB) - Eq. 25 constant m0
M0_MB = 5.0

# -----------------------------
# Reward Weights (Eq. 56)
# -----------------------------
W1_THROUGHPUT = 1.0
W2_LATENCY = 2.0
W3_CPU_WASTE = 1.0
W4_MEM_VIO = 5.0

# -----------------------------
# State Normalization
# -----------------------------
Q_MAX = 500  # Normalization constant for queue lengths (Eq. 62)

# -----------------------------
# ✅ DQN training hyperparameters (FASTER)
# -----------------------------
DQN_GAMMA = 0.99
DQN_LR = 1e-3

# smaller batches are much faster (especially on CPU)
DQN_BATCH_SIZE = 64

# smaller replay buffer = less memory + slightly faster sampling
DQN_REPLAY_SIZE = 50_000

# learn earlier (big speed win)
DQN_MIN_REPLAY_TO_LEARN = 1_000

# train less often (big speed win)
DQN_TRAIN_EVERY = 8            # was 1

# update target less often (speed + stability)
DQN_TARGET_UPDATE_EVERY = 5_000  # was 500

# Epsilon-greedy exploration schedule (faster decay)
DQN_EPS_START = 1.0
DQN_EPS_END = 0.05
DQN_EPS_DECAY_STEPS = 80_000     # was 200_000

# Gradient stability
DQN_GRAD_CLIP_NORM = 10.0

# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_SEED = 42

REWARD_WV = 1.0
REWARD_WR = 1.0
REWARD_ALPHA = 2.0      # latency weight (increase to 2 or 3 to stop queue explosion)
REWARD_GAMMA = 1.0      # constraint penalty weight (keep 1.0)

# Fixed latency normalization (seconds) for stable learning
# (prevents "dynamic max" from hiding huge absolute latency)
LV_NORM_SEC = 10.0
LR_NORM_SEC = 10.0

# -----------------------------
# Logging & Reproducibility
# -----------------------------
LOG_DIR = "logs"
SAVE_SNAPSHOTS = True
METRICS_REPORT_INTERVAL = 10  # log aggregate every N episodes
