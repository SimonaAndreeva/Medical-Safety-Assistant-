import sys
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sqlalchemy import create_engine, text

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from src.config import settings
from src.algorithms.rwr import RWR

class HINService:
    """
    The 'Advanced' Brain.
    Uses the Heterogeneous Information Network (Drugs + Proteins) to find connections.
    """
    
    def __init__(self):
        self.transition_matrix = None
        self.mapping = None
        self.engine = None
        
    def load_data(self):
        """Loads the massive Supra-Adjacency Matrix and ID mappings."""
        print("📥 HIN Service: Loading Heterogeneous Network...")
        
        # Paths
        hin_path = settings.HIN_FEATURES
        map_path = os.path.join(os.path.dirname(settings.HIN_FEATURES), 'hin_mapping.pkl')
        
        if not os.path.exists(hin_path) or not os.path.exists(map_path):
            print(f"❌ Error: HIN files not found at {hin_path}")
            print("   -> Run 'src/pipelines/build_hin_network.py' first.")
            return False

        # Load Matrix
        self.transition_matrix = load_npz(hin_path)
        
        # Load Mappings
        with open(map_path, 'rb') as f:
            self.mapping = pickle.load(f)
            
        # Connect to DB (for Name <-> ID resolution)
        self.engine = create_engine(settings.DB_URL)
        
        print(f"✅ HIN Service: Loaded {self.transition_matrix.shape[0]} nodes.")
        return True

    def get_similar_drugs(self, drug_name, top_n=10):
        """
        Runs RWR on the HIN graph to find biological & chemical cousins.
        """
        if self.transition_matrix is None:
            print("⚠️ HIN models not loaded.")
            return [], "Models not loaded"

        # 1. Resolve Name -> DB ID
        drug_id, real_name = self._resolve_name_to_id(drug_name)
        if not drug_id:
            return [], f"Drug '{drug_name}' not found in database."

        # 2. Map DB ID -> Matrix Index
        if drug_id not in self.mapping['drug_to_idx']:
            return [], f"Drug '{real_name}' (ID: {drug_id}) is not in the network (likely no chemical data)."
        
        start_node_idx = self.mapping['drug_to_idx'][drug_id]

        # 3. Run Random Walk with Restart (RWR)
        # Create a restart vector with 1.0 at the drug's position
        n_nodes = self.transition_matrix.shape[0]
        p0 = np.zeros(n_nodes)
        p0[start_node_idx] = 1.0
        
        # === USE HIN SPECIFIC SETTINGS ===
        rwr_scores = RWR.calculate_rwr(
            self.transition_matrix, 
            p0, 
            restart_prob=settings.HIN_RWR_RESTART_PROB, # Uses 0.15
            max_iter=settings.HIN_RWR_MAX_ITER,
            tol=settings.HIN_RWR_TOLERANCE
        )

        # 4. Extract Top Results (Drugs Only)
        n_drugs = self.mapping['n_drugs']
        drug_scores = rwr_scores[:n_drugs]
        
        # Get Top N indices
        top_indices = np.argsort(drug_scores)[::-1][:top_n+1]
        
        results = []
        target_ids = []
        
        # Collect IDs to fetch names in bulk
        temp_results = []
        for idx in top_indices:
            score = drug_scores[idx]
            target_id = self.mapping['drug_ids'][idx]
            
            if target_id == drug_id: continue # Skip itself
            
            temp_results.append((target_id, score))
            target_ids.append(target_id)

        # 5. Resolve Result IDs -> Names
        names_map = self._resolve_ids_to_names(target_ids)
        
        for tid, score in temp_results:
            results.append({
                'id': tid,
                'name': names_map.get(tid, f"Unknown ID {tid}"),
                'score': float(score),
                'type': 'HIN Network'
            })
            
        return results[:top_n], None

    def _resolve_name_to_id(self, name):
        """Helper: Queries SQL to get ID from Name."""
        query = text("SELECT id, generic_name FROM drugs WHERE generic_name ILIKE :name LIMIT 1")
        with self.engine.connect() as conn:
            result = conn.execute(query, {"name": name}).fetchone()
            if result:
                return result[0], result[1]
        return None, None

    def _resolve_ids_to_names(self, id_list):
        """Helper: Bulk fetches names for a list of IDs."""
        if not id_list: return {}
        query = text("SELECT id, generic_name FROM drugs WHERE id IN :ids")
        with self.engine.connect() as conn:
            results = conn.execute(query, {"ids": tuple(id_list)}).fetchall()
        return {row[0]: row[1] for row in results}