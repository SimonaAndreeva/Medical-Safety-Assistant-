import sys
import os

# --- PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.append(project_root)

import pandas as pd
import pickle
from sqlalchemy import create_engine

# Import from the clean structure
from src.config import settings
from src.models.tier_1_similarity.chemical_sim import SimilarityEngine

TEST_DRUG = "ibuprofen"

def test_similarity():
    print(f"Testing Similarity Engine for: {TEST_DRUG.upper()}")

    # 1. Load Data
    feature_file = settings.CHEMICAL_FEATURES
    print(f"   -> Loading features from: {feature_file}")
    
    try:
        with open(feature_file, 'rb') as f:
            df_features = pickle.load(f)
    except FileNotFoundError:
        print("Error: Feature file not found. Did you run 'src/features/build_chemical.py'?")
        return
    
    # 2. Connect to DB
    engine = create_engine(settings.DB_URL)
    df_names = pd.read_sql("SELECT id, generic_name FROM drugs", engine)
    
    name_to_id = {name.lower(): id_ for name, id_ in zip(df_names['generic_name'], df_names['id']) if name}
    id_to_name = dict(zip(df_names['id'], df_names['generic_name']))

    target_id = name_to_id.get(TEST_DRUG.lower())
    if not target_id: 
        print("Drug not found.")
        return

    # 3. Calculate Similarity
    if target_id not in df_features.index:
        print("Drug found in DB but has no chemical features.")
        return

    target_vector = df_features.loc[target_id].values.reshape(1, -1)
    
    # Using the math util
    scores = SimilarityEngine.calculate_tanimoto(target_vector, df_features.values)

    # 4. Show Results
    results = list(zip(df_features.index, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nRESULTS: Top 10 Chemically Similar")
    print("-" * 50)
    print(f"{'Drug Name':<30} | {'Tanimoto Score'}")
    print("-" * 50)

    for drug_id, score in results[:11]:
        name = id_to_name.get(drug_id, "Unknown")
        print(f"{name:<30} | {score:.4f}")

if __name__ == "__main__":
    test_similarity()