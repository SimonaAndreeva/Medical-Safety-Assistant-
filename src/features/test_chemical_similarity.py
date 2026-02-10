import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# --- CONFIGURATION ---
DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"
FEATURE_FILE = "data/processed/chemical_fingerprints.pkl"
TEST_DRUG = "ibuprofen"

def calculate_tanimoto(target_vector, all_vectors):
    # Convert to boolean (True/False)
    t_bool = target_vector.astype(bool)
    all_bool = all_vectors.astype(bool)
    
    # Intersection (Both have 1)
    intersection = np.sum(all_bool & t_bool, axis=1)
    
    # Union (Either has 1)
    sum_target = np.sum(t_bool)
    sum_all = np.sum(all_bool, axis=1)
    union = sum_target + sum_all - intersection
    
    # Avoid division by zero
    union[union == 0] = 1 
    
    return intersection / union

def test_similarity():
    print(f"🔎 Testing Tanimoto for: {TEST_DRUG.upper()}")

    # Load Data
    with open(FEATURE_FILE, 'rb') as f:
        df_features = pickle.load(f)
    
    engine = create_engine(DB_URL)
    df_names = pd.read_sql("SELECT id, generic_name FROM drugs", engine)
    
    # Create dictionary
    name_to_id = {name.lower(): id_ for name, id_ in zip(df_names['generic_name'], df_names['id']) if name}
    id_to_name = dict(zip(df_names['id'], df_names['generic_name']))

    target_id = name_to_id.get(TEST_DRUG.lower())
    if not target_id: return

    # Calculate Tanimoto
    target_vector = df_features.loc[target_id].values.reshape(1, -1)
    scores = calculate_tanimoto(target_vector, df_features.values).flatten()

    # Sort
    results = list(zip(df_features.index, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nRESULTS: Top 10 Chemically Similar (Tanimoto)")
    print("-" * 50)
    print(f"{'Drug Name':<30} | {'Tanimoto Score'}") # <--- Look for this header!
    print("-" * 50)

    for drug_id, score in results[:11]:
        name = id_to_name.get(drug_id, "Unknown")
        print(f"{name:<30} | {score:.4f}")

if __name__ == "__main__":
    test_similarity()