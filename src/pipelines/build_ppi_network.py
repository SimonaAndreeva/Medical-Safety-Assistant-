import sys
import os
import time

# --- PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
# --------------------------

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine

from src.config import settings
from src.models.tier_2_network.rwr import RWR

def build_network_features():
    print("🧬 Starting Advanced Biological Network Builder (RWR)...")
    start_time = time.time()

    # 1. Connect to DB
    engine = create_engine(settings.DB_URL)

    # 2. Load Data
    print("   -> Fetching Protein Interactions (PPI) and Drug Targets...")
    df_ppi = pd.read_sql("SELECT protein_a_uniprot, protein_b_uniprot FROM protein_interactions", engine)
    df_targets = pd.read_sql("SELECT drug_id, target_uniprot_id FROM drug_targets", engine)

    # 3. Define the "Biological Universe" (Every unique protein)
    print("   -> Mapping the Biological Universe...")
    universe = set(df_ppi['protein_a_uniprot']) \
               .union(set(df_ppi['protein_b_uniprot'])) \
               .union(set(df_targets['target_uniprot_id']))
    
    universe = sorted(list(universe))
    N = len(universe)
    prot_to_idx = {prot: idx for idx, prot in enumerate(universe)}
    print(f"      * Total Unique Proteins (Nodes): {N}")

    # 4. Build Adjacency Matrix
    print("   -> Building Sparse Adjacency Matrix...")
    row_idx = [prot_to_idx[p] for p in df_ppi['protein_a_uniprot']]
    col_idx = [prot_to_idx[p] for p in df_ppi['protein_b_uniprot']]
    
    # PPIs are undirected: if A interacts with B, B interacts with A
    rows = np.array(row_idx + col_idx)
    cols = np.array(col_idx + row_idx)
    data = np.ones(len(rows))

    adj_matrix = csr_matrix((data, (rows, cols)), shape=(N, N))
    print(f"      * Total Edges Mapped: {adj_matrix.nnz}")

    # 5. Build Transition Matrix for RWR
    print("   -> Converting to Column-Normalized Transition Matrix...")
    transition_matrix = RWR.build_transition_matrix(adj_matrix)

    # 6. Run Random Walk with Restart for Every Drug
    print("   -> Simulating Random Walks for Drugs (This may take a moment)...")
    unique_drugs = df_targets['drug_id'].unique()
    total_drugs = len(unique_drugs)
    
    rwr_features = {}
    
    for i, drug_id in enumerate(unique_drugs):
        # Print progress every 100 drugs
        if (i + 1) % 100 == 0 or i == 0:
            print(f"      * Processing drug {i + 1} / {total_drugs}...")

        # Find direct targets
        targets = df_targets[df_targets['drug_id'] == drug_id]['target_uniprot_id']
        
        # Create initial probability vector (P_0)
        p0 = np.zeros(N)
        for t in targets:
            if t in prot_to_idx:
                p0[prot_to_idx[t]] = 1.0
                
        # Run the RWR Algorithm!
        p_steady = RWR.calculate_rwr(
            transition_matrix=transition_matrix, 
            initial_vector=p0, 
            restart_prob=settings.RWR_RESTART_PROB,  # <--- Now it looks here!
            max_iter=settings.RWR_MAX_ITER,          # <--- And here!
            tol=settings.RWR_TOLERANCE               # <--- And here!
        )
        
        rwr_features[drug_id] = p_steady

    # 7. Save the resulting matrix
    print(f"   -> Saving continuous RWR embeddings to: {settings.NETWORK_FEATURES}")
    df_rwr = pd.DataFrame.from_dict(rwr_features, orient='index', columns=universe)
    
    os.makedirs(os.path.dirname(settings.NETWORK_FEATURES), exist_ok=True)
    with open(settings.NETWORK_FEATURES, 'wb') as f:
        pickle.dump(df_rwr, f)

    elapsed = round(time.time() - start_time, 2)
    print(f"✅ Biological Network Module Complete! (Took {elapsed} seconds)")

if __name__ == "__main__":
    build_network_features()