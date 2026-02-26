import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Ensure src is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import settings

class PPINetworkModel:
    """
    Tier 2 — Protein-Protein Interaction (PPI) Network Embedding Model.
    
    Loads precomputed Random Walk with Restart (RWR) biological footprints 
    for drugs and computes vectorized Cosine Similarity.
    
    This replaces the slow, on-the-fly graph traversal with a lightning-fast
    O(1) embedding lookup.
    """
    def __init__(self):
        # 1. Load the precomputed dense RWR embeddings
        matrix_path = settings.NETWORK_FEATURES
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f"🚨 Precomputed network features not found at {matrix_path}.\n"
                "Please run `python src/evaluation/precompute_ppi_rwr.py` first."
            )
            
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)
            
        # 2. Extract Data
        self.drug_ids = data['drug_ids']
        self.embeddings_matrix = data['vectors']  # Shape: (Num_Drugs, Num_Proteins)
        
        # 3. Create O(1) lookup map (Drug ID -> Row Index)
        self.idx_map = {d_id: idx for idx, d_id in enumerate(self.drug_ids)}
        
        print(f"🕸️  Tier 2 PPI Model Initialized with {len(self.drug_ids)} precomputed footprints.")

    def get_network_similarity(self, query_id_list, database_id_list):
        """
        Vectorized Cosine Similarity for network embeddings.
        Supports single IDs, lists of IDs, or matching pairs (returns diagonal).
        
        Signature matches Tier 1 models (e.g., TanimotoEngine, JaccardEngine).
        """
        # Ensure inputs are lists for batch processing
        if not isinstance(query_id_list, list): query_id_list = [query_id_list]
        if not isinstance(database_id_list, list): database_id_list = [database_id_list]
        
        num_queries = len(query_id_list)
        num_database = len(database_id_list)
        
        # Map IDs to actual row indices in the dense matrix
        # If ID is missing (no targets in PPI), return -1
        query_indices = [self.idx_map.get(d_id, -1) for d_id in query_id_list]
        database_indices = [self.idx_map.get(d_id, -1) for d_id in database_id_list]
        
        # Identify missing drugs
        query_valid_mask = np.array(query_indices) != -1
        database_valid_mask = np.array(database_indices) != -1
        
        # Pre-allocate output matrix of zeros
        results_matrix = np.zeros((num_queries, num_database), dtype=np.float32)
        
        # If nobody has data, return early
        if not np.any(query_valid_mask) or not np.any(database_valid_mask):
            return self._format_output(results_matrix, num_queries, num_database)
            
        # Extract valid dense vectors
        valid_q_idx = [i for i in query_indices if i != -1]
        valid_d_idx = [i for i in database_indices if i != -1]
        
        q_vectors = self.embeddings_matrix[valid_q_idx]
        d_vectors = self.embeddings_matrix[valid_d_idx]
        
        # 🚀 Vectorized Math Engine: Cosine Similarity
        # Shape: (Valid_Queries, Valid_Database)
        cosine_scores = cosine_similarity(q_vectors, d_vectors)
        
        # Scatter results back into the padded zero-matrix
        q_mask_positions = np.where(query_valid_mask)[0]
        d_mask_positions = np.where(database_valid_mask)[0]
        
        for i, raw_q in enumerate(q_mask_positions):
            for j, raw_d in enumerate(d_mask_positions):
                results_matrix[raw_q, raw_d] = cosine_scores[i, j]
                
        return self._format_output(results_matrix, num_queries, num_database)
        
    def _format_output(self, matrix, n_q, n_d):
        """Helper to match Tier 1 return shapes (Diagonal mapping for identical pairs)."""
        if n_q == n_d and n_q > 1 and matrix.shape == (n_q, n_d):
            return np.diag(matrix)
            
        if n_q == 1 and n_d == 1:
            return matrix.flatten()[0] # Return single float
            
        return matrix.flatten() if n_q == 1 else matrix

if __name__ == "__main__":
    # Quick Test
    model = PPINetworkModel()
    
    # Let's test two drugs from the cache
    if len(model.drug_ids) >= 2:
        d1 = model.drug_ids[0]
        d2 = model.drug_ids[1]
        score = model.get_network_similarity(d1, d2)
        print(f"\n✅ Network Topology Similarity Score ({d1} vs {d2}): {score:.4f}")