import os
from pathlib import Path

# Automatic Project Root Detection
# (Finds the folder where 'src' lives)
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

class Config:
    # Database (Read from Environment or use default)
    # NOTE: We can use os.getenv to read from a real .env file later
    DB_USER = "admin" # or ml_user
    DB_PASS = "12345"
    DB_HOST = "127.0.0.1"
    DB_PORT = "5435"
    DB_NAME = "medical_safety_db"
    
    @property
    def DB_URL(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # File Paths (Centralized!)
    CHEMICAL_FEATURES = DATA_DIR / "chemical_fingerprints.pkl"
    NETWORK_FEATURES = DATA_DIR / "network_features.pkl"

# Singleton instance
settings = Config()