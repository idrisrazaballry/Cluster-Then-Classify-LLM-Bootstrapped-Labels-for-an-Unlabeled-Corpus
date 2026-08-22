"""Central configuration. Every path and hyperparameter lives here."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

# --- data ---------------------------------------------------------------
RAW_CSV = DATA / "test.csv"
TEXT_COLS = ("Title", "Description")
LABEL_COL = "Class Index"
CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
MIN_WORDS = 5

# --- embeddings ---------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH = 64

# --- clustering ---------------------------------------------------------
K_SWEEP = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
K_FINAL = 4
RANDOM_STATE = 42

# --- LLM ----------------------------------------------------------------
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0
REPS_PER_CLUSTER = 15
SPOT_CHECK_N = 300
SPOT_CHECK_BATCH = 20

# --- evaluation ---------------------------------------------------------
TEST_SIZE = 0.2
HYBRID_N_TRUE = 100          # hand-labelled rows in the hybrid ablation

# --- artifact paths -----------------------------------------------------
P_QUARANTINE = ART / "_true_labels_DO_NOT_OPEN_UNTIL_PHASE_5.csv"
P_CLEAN = ART / "phase0_clean.csv"
P_EMB = ART / "embeddings.npy"
P_UMAP = ART / "umap_2d.npy"
P_CLUSTERS = ART / "phase2_clustered.csv"
P_CLUSTER_LABELS = ART / "cluster_labels.npy"
P_HUMAN_DESC = ART / "phase2_human_descriptions.json"
P_LLM_LABELS = ART / "phase3_cluster_labels.json"
P_LLM_RAW = ART / "phase3_raw_response.txt"
P_BOOTSTRAPPED = ART / "phase3_bootstrapped.csv"
P_SPOTCHECK = ART / "phase3_spotcheck.csv"
P_SPLIT = ART / "phase4_split.npz"
P_RESULTS = ART / "phase5_results.json"
P_ABLATION = ART / "phase5_ablation.csv"
