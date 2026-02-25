import os
from pathlib import Path

# Automatic Project Root Detection
# (Finds the folder where 'src' lives)
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

class Config:
    # Database (Using your original credentials)
    DB_USER = os.getenv("POSTGRES_USER", "admin")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "12345")
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = os.getenv("DB_PORT", "5435")
    DB_NAME = os.getenv("DB_NAME", "medical_safety_db")
    
    @property
    def DB_URL(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # File Paths (Centralized!)
    # Note: We added the HIN_FEATURES path here
    CHEMICAL_FEATURES = DATA_DIR / "chemical_fingerprints.pkl"
    NETWORK_FEATURES = DATA_DIR / "network_features.pkl"
    HIN_FEATURES = DATA_DIR / "hin_transition_matrix.npz"

    # ⚙️ --- ML HYPERPARAMETERS --- ⚙️
    
    # 0. FUSION TIER WEIGHTS
    FUSION_WEIGHT_PHENO = 1.0  # Data-tuned optimal weight (prev 0.6)


    # 1. PPI-ONLY MODEL (Local Clusters)
    # High probability because we trust direct neighbors in the protein graph.
    PPI_RWR_RESTART_PROB = 0.85
    PPI_RWR_MAX_ITER = 100
    PPI_RWR_TOLERANCE = 1e-3

    # 2. HETEROGENEOUS MODEL (Global/Long Bridges)
    # Low probability (0.15) to allow the walker to cross from Drug -> Target -> Protein.
    HIN_RWR_RESTART_PROB = 0.05 
    HIN_RWR_MAX_ITER = 100
    HIN_RWR_TOLERANCE = 1e-6 

# Singleton instance
settings = Config()