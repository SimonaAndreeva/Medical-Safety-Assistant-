import sys
import os

# --- PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from src.config import settings

def generate_network_features():
    print("Connecting to database...")
    try:
        engine = create_engine(settings.DB_URL)
        # Test connection
        with engine.connect() as conn:
            pass
        print("   -> Connection successful!")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # 1. Load Data
    print("Loading Biological Data...")
    
    # Get Direct Targets (Drug -> Protein)
    df_targets = pd.read_sql("SELECT drug_id, target_uniprot_id FROM drug_targets", engine)
    
    # Get PPI Network (Protein <-> Protein)
    df_ppi = pd.read_sql("SELECT protein_a_uniprot, protein_b_uniprot FROM protein_interactions", engine)
    
    print(f"   -> Loaded {len(df_targets)} direct drug-target pairs.")
    print(f"   -> Loaded {len(df_ppi)} protein interactions.")

    # 2. Build the Network Graph (Dictionary)
    print("Building the Interactome Graph...")
    ppi_graph = {}
    
    for _, row in df_ppi.iterrows():
        p1, p2 = row['protein_a_uniprot'], row['protein_b_uniprot']
        
        # Add connection (Bidirectional)
        if p1 not in ppi_graph: ppi_graph[p1] = set()
        if p2 not in ppi_graph: ppi_graph[p2] = set()
        
        ppi_graph[p1].add(p2)
        ppi_graph[p2].add(p1)

    # 3. Define the "Universe"
    # The columns of the matrix will be EVERY unique protein involved
    all_target_proteins = set(df_targets['target_uniprot_id'].unique())
    all_ppi_proteins = set(ppi_graph.keys())
    
    # Combined list of all unique proteins
    feature_universe = sorted(list(all_target_proteins.union(all_ppi_proteins)))
    protein_to_index = {prot: i for i, prot in enumerate(feature_universe)}
    
    print(f"   -> The Biological Universe consists of {len(feature_universe)} proteins.")

    # 4. Generate Drug Network Vectors (Propagation)
    print("Calculating Network Propagation (Expanding targets to neighbors)...")
    
    # Group targets by drug
    drug_groups = df_targets.groupby('drug_id')['target_uniprot_id'].apply(list)
    
    network_vectors = []
    valid_ids = []

    count = 0
    # For every drug
    for drug_id, targets in drug_groups.items():
        # Start with all zeros
        vec = np.zeros(len(feature_universe), dtype=int)
        
        # Set of "Affected" proteins
        affected_proteins = set(targets) # Start with Direct Targets
        
        # Add 1st-Order Neighbors (The "Network" part)
        for target in targets:
            if target in ppi_graph:
                # Add everyone this target talks to
                affected_proteins.update(ppi_graph[target])
        
        # Mark these proteins as "1" in the vector
        for prot in affected_proteins:
            if prot in protein_to_index:
                idx = protein_to_index[prot]
                vec[idx] = 1
                
        network_vectors.append(vec)
        valid_ids.append(drug_id)
        
        count += 1
        if count % 500 == 0:
            print(f"      ...processed {count} drugs")

    # 5. Save the Matrix
    X_network = pd.DataFrame(network_vectors, index=valid_ids, columns=feature_universe)
    
    print(f"   -> Created Matrix: {X_network.shape} (Drugs x Proteins)")
    
    # Use output path from Config
    output_path = settings.NETWORK_FEATURES
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(X_network, f)

    print(f"Saved features to: {output_path}")
    print("Biological Network Module Complete.")

if __name__ == "__main__":
    generate_network_features()