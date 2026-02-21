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
# PDU session request model
# -----------------------------
# Data rates (kbps) + 1:2:3:4 distribution
DATA_RATES = [128, 256, 512, 1024]
DATA_RATE_RATIO = [1, 2, 3, 4]

# Operator split for users
USER_OPERATOR_RATIO = (5, 5)

# Each slot, each user generates a new session with this probability.
P_SESSION_REQUEST = 0.05

# -----------------------------
# Latency: gNB -> edge hop (distance-based transport)
# -----------------------------
# Convert gNB->edge distance into delay:
# L = (BETA_MS_PER_KM * (d_m / 1000)) / 1000  seconds
BETA_GNB_EDGE_MS_PER_KM = 2.0

# -----------------------------
# Resource budgets (your locked numbers)
# -----------------------------
EDGE_CAP_SESSIONS = 250  # used as your budget sizing knob (as you set earlier)

# ---- CPU budget (millicores) ----
CPU_IDLE_MC = 93
CPU_PER_SESSION_MC = 1.32  # millicores per session (avg)
EDGE_CPU_BUDGET_MC = int(CPU_PER_SESSION_MC * EDGE_CAP_SESSIONS)  # ~262

# ---- Memory budget (MB) ----
# PDF model uses: M_used = m0 * (#served sessions)
MEM_PER_SESSION_MB = 5 # m0 (MB per served session) - your tuning knob
EDGE_MEM_BUDGET_MB = 1280  # keep exactly as you had

# -----------------------------
# Stage-3 (PDF helpers): class-wise CPU cost
# cpuk = cpuavg * (rk / ravg)
# -----------------------------
CPU_AVG_PER_SESSION_MC = CPU_PER_SESSION_MC  # pdf cpuavg
_RAVG_DEN = sum(DATA_RATE_RATIO)
R_AVG_KBPS = sum(rk * w for rk, w in zip(DATA_RATES, DATA_RATE_RATIO)) / _RAVG_DEN

CPU_PER_SESSION_BY_RATE_MC = {rk: CPU_AVG_PER_SESSION_MC * (rk / R_AVG_KBPS) for rk in DATA_RATES}

# Convenience alias used by other modules
M0_MB = MEM_PER_SESSION_MB

# -----------------------------
# Stage-3: Symmetric Home UPF budgets (for roaming sessions)
# -----------------------------
HOME_CPU_BUDGET_MC = EDGE_CPU_BUDGET_MC
HOME_MEM_BUDGET_MB = EDGE_MEM_BUDGET_MB

# -----------------------------
# RL / DQN action space (48 discrete actions)
# a(t) = (rho_v, rho_r, pi_v) ; pi_r = 1 - pi_v
# -----------------------------
ACTION_RHO = [0.25, 0.5, 0.75, 1.0]
ACTION_PI = [0.3, 0.5, 0.7]
NUM_ACTIONS = len(ACTION_RHO) * len(ACTION_RHO) * len(ACTION_PI)  # 48

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
