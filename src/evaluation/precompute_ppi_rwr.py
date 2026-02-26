import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine

# Ensure src is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import settings
from src.models.tier_2_network.ppi_network_model import PPINetworkModel
from src.models.tier_2_network.rwr import RWR

def precompute_ppi_embeddings():
    """
    Offline script to compute the Random Walk with Restart (RWR) steady-state
    biological footprint for every drug in the database that has known targets.
    
    This precomputes the heavy graph propagation so that the real-time scoring
    can just load a numeric matrix and perform instant Cosine Similarity.
    """
    print("🚀 Starting Tier 2: PPI Network Precomputation...")
    
    # 1. Initialize the existing model to build the transition matrix
    model = PPINetworkModel(db_url=settings.DB_URL)
    
    # 2. Identify all unique drugs that have at least one target in the graph
    # (Extracting this from the drug_targets dataframe the model already loaded)
    df_targets = model.drug_targets
    
    # Filter targets that actually exist in the nodes we extracted from the PPI
    df_targets['valid'] = df_targets['target_uniprot_id'].isin(model.node_to_idx)
    valid_drugs = df_targets[df_targets['valid']]['drug_id'].unique()
    
    num_drugs = len(valid_drugs)
    print(f"✅ Found {num_drugs} drugs with valid targets in the PPI network.")
    if num_drugs == 0:
        print("❌ No valid drugs found. Cannot precompute.")
        return
        
    num_nodes = model.num_nodes
    print(f"🧬 Embedding dimension per drug: {num_nodes} (number of proteins)")
    
    # 3. Pre-allocate the massive dense matrix
    # Shape: (Num_Drugs, Num_Proteins)
    embeddings = np.zeros((num_drugs, num_nodes), dtype=np.float32)
    drug_id_list = []
    
    print("\n🚶‍♂️ Running Random Walks...")
    for i, drug_id in enumerate(tqdm(valid_drugs, desc="Computing RWR Footprints")):
        # Get the initial probability vector p0 based on the drug's targets
        p0, target_count = model.get_drug_seed_vector(drug_id)
        
        # We only process valid drugs, so target_count should be > 0
        if target_count > 0:
            # Run the math engine
            footprint = RWR.calculate_rwr(
                transition_matrix=model.transition_matrix,
                initial_vector=p0,
                restart_prob=settings.PPI_RWR_RESTART_PROB,
                max_iter=settings.PPI_RWR_MAX_ITER,
                tol=settings.PPI_RWR_TOLERANCE
            )
            embeddings[i, :] = footprint.astype(np.float32)
            
        drug_id_list.append(drug_id)
        
    # 4. Save to disk
    output_path = settings.NETWORK_FEATURES
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    payload = {
        'drug_ids': drug_id_list,
        'vectors': embeddings,
        'metadata': {
            'num_nodes': num_nodes,
            'rwr_restart_prob': settings.PPI_RWR_RESTART_PROB
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)
        
    print(f"\n💾 Saved {num_drugs} continuous graph embeddings to {output_path}")
    
    # Print a tiny sample to prove it worked
    nz_sample = np.count_nonzero(embeddings[0])
    print(f"Sample Drug {drug_id_list[0]}: {nz_sample} non-zero protein probabilities.")

if __name__ == "__main__":
    precompute_ppi_embeddings()
