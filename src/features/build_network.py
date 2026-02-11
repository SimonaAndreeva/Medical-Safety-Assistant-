import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import os

# --- CONFIGURATION ---
# Use the connection that works for you (Port 5435)
DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"
OUTPUT_FILE = "data/processed/network_features.pkl"

def generate_network_features():
    print("Connecting to database...")
    engine = create_engine(DB_URL)

    # 1. Load Data
    print("Loading Biological Data...")
    
    # Get Direct Targets (Drug -> Protein)
    df_targets = pd.read_sql("SELECT drug_id, target_uniprot_id FROM drug_targets", engine)
    
    # Get PPI Network (Protein <-> Protein)
    df_ppi = pd.read_sql("SELECT protein_a_uniprot, protein_b_uniprot FROM protein_interactions", engine)
    
    print(f"   -> Loaded {len(df_targets)} direct drug-target pairs.")
    print(f"   -> Loaded {len(df_ppi)} protein interactions.")

    # 2. Build the Network Graph (Dictionary)
    # This allows us to quickly look up neighbors: "Who does Protein A talk to?"
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
    # The columns of our matrix will be EVERY unique protein involved
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
    # For every drug...
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
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(X_network, f)

    print(f"Saved features to: {OUTPUT_FILE}")
    print("Biological Network Module Complete.")

if __name__ == "__main__":
    generate_network_features()