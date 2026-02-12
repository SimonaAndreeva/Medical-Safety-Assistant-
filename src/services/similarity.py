import pandas as pd
import pickle
from sqlalchemy import create_engine
from src.config import settings

# --- FIX: Import from the new 'utils' folder ---
from src.utils.math import SimilarityEngine 

class DrugSimilarityService:
    def __init__(self):
        self.chemical_matrix = None
        self.network_matrix = None
        self.drug_names = {}  # ID -> Name
        self.drug_ids = {}    # Name -> ID
        self._is_loaded = False

    def load_data(self):
        """Loads all models into memory (Run this ONCE at startup)"""
        print("📥 Service: Loading AI Models...")
        
        try:
            # 1. Load Matrices
            with open(settings.CHEMICAL_FEATURES, 'rb') as f:
                self.chemical_matrix = pickle.load(f)
                
            with open(settings.NETWORK_FEATURES, 'rb') as f:
                self.network_matrix = pickle.load(f)

            # 2. Load Names from DB
            engine = create_engine(settings.DB_URL)
            df = pd.read_sql("SELECT id, generic_name FROM drugs", engine)
            
            # Create fast lookups
            self.drug_names = dict(zip(df['id'], df['generic_name']))
            self.drug_ids = {name.lower(): id_ for name, id_ in zip(df['generic_name'], df['id']) if name}
            
            self._is_loaded = True
            print("✅ Service: AI Ready.")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("   (Did you run 'build_chemical.py' and 'build_network.py'?)")

    def get_similar_drugs(self, drug_name: str, method="chemical", top_n=10):
        if not self._is_loaded:
            return None, "AI Service not loaded! Call load_data() first."

        drug_clean = drug_name.lower().strip()
        
        # 1. Check if drug exists
        if drug_clean not in self.drug_ids:
            return None, f"Drug '{drug_name}' not found in database."

        drug_id = self.drug_ids[drug_clean]
        
        # 2. Select the correct matrix
        if method == "chemical":
            matrix = self.chemical_matrix
        elif method == "network":
            matrix = self.network_matrix
        else:
            return None, "Invalid method (use 'chemical' or 'network')"

        # 3. Check if drug has features in that matrix
        if drug_id not in matrix.index:
            return None, f"Drug exists, but has no {method} features (structure/targets missing)."

        # 4. Calculate Similarity (The Math!)
        target_vec = matrix.loc[drug_id].values.reshape(1, -1)
        all_vecs = matrix.values
        
        # Use the Utils Engine
        scores = SimilarityEngine.calculate_tanimoto(target_vec, all_vecs).flatten()

        # 5. Format Results
        results = []
        for idx, score in sorted(zip(matrix.index, scores), key=lambda x: x[1], reverse=True)[:top_n+1]:
            # Skip the drug itself (score = 1.0)
            if idx == drug_id:
                continue
                
            results.append({
                "name": self.drug_names.get(idx, "Unknown"),
                "score": round(score, 4)
            })
            
        return results, None
    