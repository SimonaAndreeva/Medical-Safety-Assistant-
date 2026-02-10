import pandas as pd
import pickle
from sqlalchemy import create_engine
# IMPORT YOUR NEW LIBRARY HERE:
from src.features.similarity_utils import SimilarityEngine

# --- CONFIGURATION ---
DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"
FEATURE_FILE = "data/processed/chemical_fingerprints.pkl"
TEST_DRUG = "ibuprofen"

def test_similarity():
    print(f"Testing Similarity Engine for: {TEST_DRUG.upper()}")

    # 1. Load Data
    with open(FEATURE_FILE, 'rb') as f:
        df_features = pickle.load(f)
    
    engine = create_engine(DB_URL)
    df_names = pd.read_sql("SELECT id, generic_name FROM drugs", engine)
    
    name_to_id = {name.lower(): id_ for name, id_ in zip(df_names['generic_name'], df_names['id']) if name}
    id_to_name = dict(zip(df_names['id'], df_names['generic_name']))

    target_id = name_to_id.get(TEST_DRUG.lower())
    if not target_id: 
        print("Drug not found.")
        return

    # 2. CALL THE REUSABLE ALGORITHM
    target_vector = df_features.loc[target_id].values.reshape(1, -1)
    all_vectors = df_features.values
    
    scores = SimilarityEngine.calculate_tanimoto(target_vector, all_vectors)

    # 3. Show Results
    results = list(zip(df_features.index, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nRESULTS: Top 10 Chemically Similar (Refactored)")
    print("-" * 50)
    print(f"{'Drug Name':<30} | {'Tanimoto Score'}")
    print("-" * 50)

    for drug_id, score in results[:11]:
        name = id_to_name.get(drug_id, "Unknown")
        print(f"{name:<30} | {score:.4f}")

if __name__ == "__main__":
    test_similarity()