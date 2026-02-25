import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys

# --- PATHS & CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import settings
from src.models.tier_1_similarity.chemical_sim import SimilarityEngine
PROCESSED_DATA_DIR = "data/processed/"
PHENO_MATRIX_PATH = os.path.join(PROCESSED_DATA_DIR, "phenotypic_matrix.pkl")

class AdvancedFusionModel:
    def __init__(self):
        # Load the side-effect matrix we just built
        with open(PHENO_MATRIX_PATH, 'rb') as f:
            df_pheno = pickle.load(f) # Rows=DrugIDs, Cols=SideEffects
        
        # O(1) Hash Map for row lookup
        self.known_pheno_ids = set(df_pheno.index)
        self.pheno_idx_map = {drug_id: idx for idx, drug_id in enumerate(df_pheno.index)}
        
        # Convert heavy Pandas DataFrame to fast Sparse CSR Matrix
        self.pheno_matrix_sparse = csr_matrix(df_pheno.values)
        
        self._fp_cache = {}  # Upgrade 2: Fingerprint Cache
        print(f"🧬 Fusion Model Initialized with {len(self.known_pheno_ids)} phenotypic drug profiles (Sparse Vectorized).")

    def _get_fingerprint(self, smiles):
        """Helper to compute and cache Morgan fingerprints as flat numpy bit arrays."""
        if not smiles or not isinstance(smiles, str):
            return np.zeros(2048, dtype=int)
        
        if smiles in self._fp_cache:
            return self._fp_cache[smiles]
            
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            zeros = np.zeros(2048, dtype=int)
            self._fp_cache[smiles] = zeros
            return zeros
            
        # Extract RDKit Object and immediately convert to native NumPy Array (1s and 0s)
        fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_obj, fp_arr)
        
        self._fp_cache[smiles] = fp_arr
        return fp_arr

    def get_chemical_similarity(self, query_smiles_list, database_smiles_list):
        """Vectorized Tanimoto similarity for 2D structures."""
        # Ensure inputs are lists for batch processing
        if isinstance(query_smiles_list, str): query_smiles_list = [query_smiles_list]
        if isinstance(database_smiles_list, str): database_smiles_list = [database_smiles_list]
        
        # Build completely stacked 2D Feature Matrices
        query_fingerprints = np.vstack([self._get_fingerprint(s) for s in query_smiles_list])
        database_fingerprints = np.vstack([self._get_fingerprint(s) for s in database_smiles_list])
        
        # Delegate to the hyper-optimized Tier 1 Sparse CSR engine
        # Calculate diagonal (pairwise Q[i] vs D[i]) if they are same length lists
        # If lengths differ, calculate_tanimoto will return the full M x N interaction matrix
        full_matrix = SimilarityEngine.calculate_tanimoto(query_fingerprints=query_fingerprints, database_fingerprints=database_fingerprints)
        
        # If the requester passed exactly matched lists (e.g. for evaluating pairs),
        # extract just the diagonal Q[0]vsD[0], Q[1]vsD[1]...
        if len(query_smiles_list) == len(database_smiles_list) and len(query_smiles_list) > 1 and full_matrix.shape == (len(query_smiles_list), len(database_smiles_list)):
             return np.diag(full_matrix)
        
        return full_matrix.flatten() if len(query_smiles_list) == 1 and len(database_smiles_list) == 1 else full_matrix

    def get_phenotypic_similarity(self, query_id_list, database_id_list):
        """Vectorized Jaccard similarity for side-effect overlap using CSR matrices."""
        if isinstance(query_id_list, str): query_id_list = [query_id_list]
        if isinstance(database_id_list, str): database_id_list = [database_id_list]
        
        num_queries = len(query_id_list)
        num_database = len(database_id_list)
        
        # Map IDs to physical row indices, isolating missing IDs as -1
        query_indices = [self.pheno_idx_map.get(i, -1) for i in query_id_list]
        database_indices = [self.pheno_idx_map.get(i, -1) for i in database_id_list]
        
        # Identify which drugs actually have phenotypic data
        query_valid_mask = np.array(query_indices) != -1
        database_valid_mask = np.array(database_indices) != -1
        
        # Pre-allocate output matrix of zeros
        results_matrix = np.zeros((num_queries, num_database))
        
        if not np.any(query_valid_mask) or not np.any(database_valid_mask):
            return np.zeros(num_queries) if (num_queries == num_database and num_queries > 1) else (results_matrix.flatten() if num_queries==1 and num_database==1 else results_matrix)

        # Extract only the valid rows into sliced CSR matrices
        query_valid_indices = [i for i in query_indices if i != -1]
        database_valid_indices = [i for i in database_indices if i != -1]
        
        query_sparse = self.pheno_matrix_sparse[query_valid_indices]
        database_sparse = self.pheno_matrix_sparse[database_valid_indices]
        
        # Mathematical Intersection & Union 
        # (Same optimal logic as chemical_sim, tailored for the pheno matrix)
        shared_phenotypes_count = query_sparse.dot(database_sparse.T).toarray()
        
        query_phenotypes_count = np.array(query_sparse.sum(axis=1))
        database_phenotypes_count = np.array(database_sparse.sum(axis=1)).T
        
        total_unique_phenotypes_count = query_phenotypes_count + database_phenotypes_count - shared_phenotypes_count
        total_unique_phenotypes_count[total_unique_phenotypes_count == 0] = 1.0 # Prevent zero division
        
        jaccard_scores = shared_phenotypes_count / total_unique_phenotypes_count
        
        # Scatter valid results back into the padded zero-matrix
        query_mask_positions = np.where(query_valid_mask)[0]
        database_mask_positions = np.where(database_valid_mask)[0]
        
        for i, q_raw_idx in enumerate(query_mask_positions):
            for j, d_raw_idx in enumerate(database_mask_positions):
                results_matrix[q_raw_idx, d_raw_idx] = jaccard_scores[i, j]
                
        # Return diagonal if matched lists, else full matrix
        if num_queries == num_database and num_queries > 1 and results_matrix.shape == (num_queries, num_database):
             return np.diag(results_matrix)
             
        return results_matrix.flatten() if num_queries == 1 and num_database == 1 else results_matrix

    def predict_fusion_score(self, query_drugs, database_drugs, weight_pheno=None):
        """
        Vectorized Fusion.
        query_drugs and database_drugs can be single dictionaries or lists of dictionaries.
        Each dict must have 'smiles' and 'id' keys.
        """
        is_single_comparison = isinstance(query_drugs, dict)
        if is_single_comparison:
            query_drugs = [query_drugs]
            database_drugs = [database_drugs]
            
        query_smiles = [d['smiles'] for d in query_drugs]
        database_smiles = [d['smiles'] for d in database_drugs]
        query_ids = [d['id'] for d in query_drugs]
        database_ids = [d['id'] for d in database_drugs]
        
        # Compute heavy matrices in bulk using purely vectorized core modules
        chemical_scores = self.get_chemical_similarity(query_smiles, database_smiles)
        phenotypic_scores = self.get_phenotypic_similarity(query_ids, database_ids)
        
        if weight_pheno is None:
            weight_pheno = getattr(settings, 'FUSION_WEIGHT_PHENO', 1.0)
            
        # Dynamically switch to chemical similarity wherever phenotypic data is fully missing (score is exactly 0)
        final_scores = np.where(phenotypic_scores == 0, chemical_scores, (weight_pheno * phenotypic_scores) + ((1 - weight_pheno) * chemical_scores))
        
        return final_scores[0] if is_single_comparison else final_scores

# Example Usage logic
if __name__ == "__main__":
    # This would normally be fed from your database
    drug_a = {'id': 1, 'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'} # Aspirin
    drug_b = {'id': 2, 'smiles': 'CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC'} # Cocaine
    
    model = AdvancedFusionModel()
    score = model.predict_fusion_score(drug_a, drug_b)
    print(f"🔥 Combined Fusion Similarity Score: {score:.4f}")