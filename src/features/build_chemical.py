import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os

# --- CONFIGURATION ---
DB_URL = "postgresql://ml_user:ml123@127.0.0.1:5435/medical_safety_db"

OUTPUT_FILE = "data/processed/chemical_fingerprints.pkl"

def generate_fingerprints():
    print("🔌 Connecting to database...")
    try:
        engine = create_engine(DB_URL)
        # Test connection
        with engine.connect() as conn:
            pass
        print("   -> Connection successful!")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # 1. Fetch Drugs (ID and SMILES)
    print("Fetching drugs...")
    query = "SELECT id, smiles FROM drugs WHERE smiles IS NOT NULL"
    df = pd.read_sql(query, engine)
    print(f"   -> Loaded {len(df)} drugs with structures.")

    # 2. Convert SMILES to Fingerprints
    print("Generating Morgan Fingerprints (Radius=2, Bits=2048)...")
    
    fingerprints = []
    valid_ids = []
    
    for index, row in df.iterrows():
        smiles = row['smiles']
        drug_id = row['id']
        
        # Create Molecule
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            # Generate 2048-bit vector
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            
            # Convert to list (Standard Python way, safest for simple ML)
            fp_list = list(fp)
            
            fingerprints.append(fp_list)
            valid_ids.append(drug_id)
    
    # 3. Create DataFrame
    X_chemical = pd.DataFrame(fingerprints, index=valid_ids)
    
    print(f"   -> Created Matrix: {X_chemical.shape} (Drugs x Features)")

    # 4. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(X_chemical, f)
        
    print(f"Saved to: {OUTPUT_FILE}")
    print("Chemical Similarity Module Complete.")

if __name__ == "__main__":
    generate_fingerprints()