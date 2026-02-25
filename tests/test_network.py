import sys
import os

# --- PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
# --------------------------

import pandas as pd
import pickle
from sqlalchemy import create_engine

from src.config import settings
from src.models.tier_2_network.rwr import RWR

TEST_DRUG = "sildenafil" # Viagra

def test_network_similarity():
    print(f"🔎 Testing RWR Biological Network for: {TEST_DRUG.upper()}")

    # 1. Load Data
    feature_file = settings.NETWORK_FEATURES
    try:
        with open(feature_file, 'rb') as f:
            df_features = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: Feature file not found. Run 'build_network.py' first.")
        return

    # 2. Connect to DB
    engine = create_engine(settings.DB_URL)
    df_names = pd.read_sql("SELECT id, generic_name FROM drugs", engine)
    
    name_to_id = {name.lower(): id_ for name, id_ in zip(df_names['generic_name'], df_names['id']) if name}
    id_to_name = dict(zip(df_names['id'], df_names['generic_name']))

    # 3. Find the Drug
    target_id = name_to_id.get(TEST_DRUG.lower())
    if not target_id or target_id not in df_features.index:
        print(f"❌ '{TEST_DRUG}' not found or has no targets.")
        return

    # 4. Calculate Similarity (NOW USING COSINE!)
    target_vector = df_features.loc[target_id].values.reshape(1, -1)
    
    # -> CHANGED THIS LINE:
    scores = RWR.calculate_cosine(target_vector, df_features.values).flatten()

    # 5. Show Results
    results = list(zip(df_features.index, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n✅ RESULTS: Top 10 Biologically Similar Drugs (RWR + Cosine)")
    print("-" * 60)
    for drug_id, score in results[:11]:
        name = id_to_name.get(drug_id, "Unknown")
        print(f"{name:<30} | {score:.4f}")

if __name__ == "__main__":
    test_network_similarity()