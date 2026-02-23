import pandas as pd
import numpy as np
import gzip
import pickle
from sqlalchemy import create_engine

# Verified Connection
DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"
SIDER_PATH = "data/raw/meddra_all_se.tsv.gz"

def build_phenotype_matrix():
    engine = create_engine(DB_URL)
    
    # 1. Get drugs and their PubChem CIDs
    print("📋 Fetching drugs from database...")
    drugs = pd.read_sql("SELECT id, pubchem_cid FROM drugs WHERE pubchem_cid IS NOT NULL", engine)
    
    # SIDER uses STITCH IDs. Format: CID1000002244 (for CID 2244)
    # We map your CID to the SIDER format
    drugs['stitch_id'] = drugs['pubchem_cid'].apply(lambda x: f"CID1{int(x):08d}")
    stitch_to_id = dict(zip(drugs['stitch_id'], drugs['id']))
    
    # 2. Parse SIDER File
    print("📖 Parsing SIDER side effects (this may take a minute)...")
    se_data = []
    with gzip.open(SIDER_PATH, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            stitch_id = parts[0]
            side_effect = parts[5] # MedDRA Term
            
            if stitch_id in stitch_to_id:
                se_data.append({
                    'drug_id': stitch_to_id[stitch_id],
                    'side_effect': side_effect
                })

    df_se = pd.DataFrame(se_data).drop_duplicates()
    print(f"✅ Found {len(df_se)} drug-side effect associations.")

    # 3. Pivot to Binary Matrix (Rows=Drugs, Cols=Side Effects)
    print("🧪 Generating binary matrix...")
    matrix = pd.crosstab(df_se['drug_id'], df_se['side_effect'])
    
    # Save the processed matrix
    output_path = "data/processed/phenotypic_matrix.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(matrix, f)
    
    print(f"💾 Matrix saved to {output_path}. Shape: {matrix.shape}")

if __name__ == "__main__":
    build_phenotype_matrix()
    