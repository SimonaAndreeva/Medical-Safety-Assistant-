import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import bmat, csr_matrix, save_npz, lil_matrix
from sqlalchemy import create_engine, text

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from src.config import settings
from src.models.tier_2_network.rwr import RWR
from src.models.tier_1_similarity.chemical_sim import SimilarityEngine

def build_heterogeneous_network():
    print("🧬 STARTING PIPELINE: Build Heterogeneous Information Network (HIN)...")

    # 1. Load Chemical Data
    print("   -> Loading Chemical Similarity Matrix...")
    if not os.path.exists(settings.CHEMICAL_FEATURES):
        print("❌ Error: Chemical features not found.")
        return

    with open(settings.CHEMICAL_FEATURES, 'rb') as f:
        df_chem = pickle.load(f)
    
    drug_ids = df_chem.index.tolist()
    n_drugs = len(drug_ids)
    print(f"   -> Loaded {n_drugs} Drugs.")

    # 2. Load PPI Data
    print("   -> Loading Protein Interactions...")
    engine = create_engine(settings.DB_URL)
    with engine.connect() as conn:
        prot_query = text("SELECT DISTINCT protein_a_uniprot FROM protein_interactions UNION SELECT DISTINCT protein_b_uniprot FROM protein_interactions")
        proteins = pd.read_sql(prot_query, conn)['protein_a_uniprot'].tolist()
        
        ppi_query = text("SELECT protein_a_uniprot, protein_b_uniprot FROM protein_interactions")
        df_ppi = pd.read_sql(ppi_query, conn)

        # CORRECTED QUERY (Direct Select)
        dt_query = text("SELECT drug_id as id, target_uniprot_id as uniprot_id FROM drug_targets WHERE target_uniprot_id IS NOT NULL")
        df_dt = pd.read_sql(dt_query, conn)

    prot_map = {p: i for i, p in enumerate(proteins)}
    n_proteins = len(proteins)
    print(f"   -> Loaded {n_proteins} Proteins and {len(df_ppi)} Interactions.")
    
    df_dt = df_dt[df_dt['id'].isin(drug_ids) & df_dt['uniprot_id'].isin(proteins)]
    print(f"   -> Found {len(df_dt)} Valid Drug-Target Bridges.")

    # ==========================================
    # 🏗️ BATCHED CONSTRUCTION (The Fix)
    # ==========================================
    print("   -> Constructing Supra-Adjacency Matrix (Batched)...")

    # BLOCK 1: Drug-Drug Similarity (A_dd)
    # We calculate this in chunks to avoid "Stuck" console
    drug_vectors = df_chem.values
    
    # Pre-allocate a sparse matrix for results to save memory
    # using LIL format is faster for incremental building
    A_dd = lil_matrix((n_drugs, n_drugs), dtype=float)
    
    batch_size = 500
    total_batches = (n_drugs // batch_size) + 1
    
    print(f"   -> Processing Chemical Similarity in {total_batches} batches:")
    
    for i in range(0, n_drugs, batch_size):
        end = min(i + batch_size, n_drugs)
        sys.stdout.write(f"\r      ... Batch {i//batch_size + 1}/{total_batches} (Drugs {i}-{end})")
        sys.stdout.flush()
        
        # Calculate chunk
        target_chunk = drug_vectors[i:end]
        sim_chunk = SimilarityEngine.calculate_tanimoto(target_chunk, drug_vectors)
        
        # Threshold and fill
        sim_chunk[sim_chunk < 0.3] = 0.0
        
        # Assign to the massive matrix
        A_dd[i:end, :] = sim_chunk

    print("\n   -> Chemical Matrix Built.")
    
    # Ensure diagonal is 1.0 (Self-loops)
    A_dd.setdiag(1.0)
    A_dd = A_dd.tocsr() # Convert to Compressed Sparse Row for speed

    # BLOCK 2: PPI (A_pp)
    G_ppi = nx.from_pandas_edgelist(df_ppi, 'protein_a_uniprot', 'protein_b_uniprot')
    G_ppi.add_nodes_from(proteins)
    A_pp = nx.adjacency_matrix(G_ppi, nodelist=proteins)

    # BLOCK 3: Drug-Protein (A_dp)
    row_ind = [drug_ids.index(d) for d in df_dt['id']]
    col_ind = [prot_map[p] for p in df_dt['uniprot_id']]
    data = np.ones(len(row_ind))
    A_dp = csr_matrix((data, (row_ind, col_ind)), shape=(n_drugs, n_proteins))
    
    # BLOCK 4: Transpose
    A_pd = A_dp.T

    # Stitch
    print("   -> Stitching Supra-Matrix...")
    supra_adj = bmat([[A_dd, A_dp], [A_pd, A_pp]], format='csr')
    print(f"   -> Final Shape: {supra_adj.shape}")

    # Normalization
    print("   -> Calculating Transition Matrix (RWR Norm)...")
    transition_matrix = RWR.build_transition_matrix(supra_adj)

    # Save
    output_dir = os.path.dirname(settings.NETWORK_FEATURES)
    matrix_path = os.path.join(output_dir, 'hin_transition_matrix.npz')
    mapping_path = os.path.join(output_dir, 'hin_mapping.pkl')
    
    save_npz(matrix_path, transition_matrix)
    
    mapping = {
        'drug_ids': drug_ids,
        'proteins': proteins,
        'drug_to_idx': {d: i for i, d in enumerate(drug_ids)},
        'prot_to_idx': {p: i + n_drugs for i, p in enumerate(proteins)},
        'n_drugs': n_drugs,
        'n_proteins': n_proteins
    }
    
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)

    print(f"✅ HIN Construction Complete! Saved to {matrix_path}")

if __name__ == "__main__":
    build_heterogeneous_network()