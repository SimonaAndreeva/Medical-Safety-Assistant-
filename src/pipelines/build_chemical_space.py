import sys
import os

# --- PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

import pandas as pd
from sqlalchemy import create_engine
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

from src.config import settings

def generate_fingerprints():
    print("Connecting to database...")
    try:
        # Use DB_URL from config.py
        engine = create_engine(settings.DB_URL)
        
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
            
            # Convert to list
            fp_list = list(fp)
            
            fingerprints.append(fp_list)
            valid_ids.append(drug_id)
    
    # 3. Create DataFrame
    X_chemical = pd.DataFrame(fingerprints, index=valid_ids)
    
    print(f"   -> Created Matrix: {X_chemical.shape} (Drugs x Features)")

    # 4. Save using the path from config
    output_path = settings.CHEMICAL_FEATURES
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(X_chemical, f)
        
    print(f"Saved to: {output_path}")
    print("Chemical Similarity Module Complete.")

if __name__ == "__main__":
    generate_fingerprints()