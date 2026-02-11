import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # This file's folder
project_root = os.path.abspath(os.path.join(current_dir, '../..')) # Go up 2 levels
sys.path.append(project_root)

import pandas as pd
import pickle
from sqlalchemy import create_engine
from src.features.similarity_utils import SimilarityEngine

# --- CONFIGURATION ---
DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"
FEATURE_FILE = os.path.join(project_root, "data/processed/network_features.pkl")
TEST_DRUG = "sildenafil" # Viagra

def test_network_similarity():
    print(f"Testing Biological Network Similarity for: {TEST_DRUG.upper()}")

    # 1. Load Data
    print(f"   -> Loading features from: {FEATURE_FILE}")
    try:
        with open(FEATURE_FILE, 'rb') as f:
            df_features = pickle.load(f)
    except FileNotFoundError:
        print("Error: Feature file not found. Did you run 'build_network.py'?")
        return

    # 2. Connect to Database (for names)
    engine = create_engine(DB_URL)
    df_names = pd.read_sql("SELECT id, generic_name FROM drugs", engine)
    
    # Create lookups
    name_to_id = {name.lower(): id_ for name, id_ in zip(df_names['generic_name'], df_names['id']) if name}
    id_to_name = dict(zip(df_names['id'], df_names['generic_name']))

    # 3. Find the Drug
    target_id = name_to_id.get(TEST_DRUG.lower())
    
    if not target_id:
        print(f"'{TEST_DRUG}' not found in the drug database.")
        return
        
    if target_id not in df_features.index:
        print(f"'{TEST_DRUG}' found, but it has no mapped targets in the network.")
        return

    # 4. Calculate Similarity
    # Using the 'Network Propagation' logic (Tanimoto on Proteins)
    target_vector = df_features.loc[target_id].values.reshape(1, -1)
    
    scores = SimilarityEngine.calculate_tanimoto(target_vector, df_features.values).flatten()

    # 5. Show Results
    results = list(zip(df_features.index, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nRESULTS: Top 10 Biologically Similar Drugs")
    print("-" * 60)
    print(f"{'Drug Name':<30} | {'Network Score':<15}")
    print("-" * 60)

    for drug_id, score in results[:11]:
        name = id_to_name.get(drug_id, "Unknown")
        print(f"{name:<30} | {score:.4f}")

    print("-" * 60)
    print("Interpretation: A score of 1.0 means identical targets.")
    print("High scores indicate drugs affecting the same protein modules.")

if __name__ == "__main__":
    test_network_similarity()