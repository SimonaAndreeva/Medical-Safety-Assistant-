import pandas as pd
import pickle
from sqlalchemy import create_engine
from src.config import settings
from src.models.tier_1_similarity.chemical_sim import SimilarityEngine
from src.models.tier_2_network.rwr import RWR

class DrugSimilarityService:
    def __init__(self):
        self.chemical_matrix = None
        self.network_matrix = None
        self.drug_ids = {}    # Name -> ID (The "Master Index")
        self.drug_names = {}  # ID -> Name
        self._is_loaded = False
        
        # 🧠 HARDCODED ALIASES
        self.manual_aliases = {
            "aspirin": "acetylsalicylic acid",
            "tylenol": "acetaminophen",
            "viagra": "sildenafil",
            "advil": "ibuprofen",
            "motrin": "ibuprofen"
        }

    def load_data(self):
        """Loads models and builds a smart name index."""
        print("📥 Service: Loading AI Models...")
        
        try:
            # 1. Load Matrices
            with open(settings.CHEMICAL_FEATURES, 'rb') as f:
                self.chemical_matrix = pickle.load(f)
            with open(settings.NETWORK_FEATURES, 'rb') as f:
                self.network_matrix = pickle.load(f)

            # 2. Load Drug Names
            engine = create_engine(settings.DB_URL)
            
            try:
                df = pd.read_sql("SELECT id, generic_name, synonyms FROM drugs", engine)
            except:
                df = pd.read_sql("SELECT id, generic_name FROM drugs", engine)
                df['synonyms'] = None

            # 3. Build the Master Index
            self.drug_names = dict(zip(df['id'], df['generic_name']))
            self.drug_ids = {name.lower(): id_ for name, id_ in zip(df['generic_name'], df['id']) if name}
            
            # Map Synonyms
            if 'synonyms' in df.columns:
                for _, row in df.iterrows():
                    if row['synonyms']:
                        syns = str(row['synonyms']).replace("|", ",").split(",")
                        for s in syns:
                            clean_s = s.strip().lower()
                            if clean_s and clean_s not in self.drug_ids:
                                self.drug_ids[clean_s] = row['id']

            # Map Manual Aliases
            for alias, target in self.manual_aliases.items():
                if target in self.drug_ids:
                    self.drug_ids[alias] = self.drug_ids[target]

            self._is_loaded = True
            print("✅ Service: AI Ready.")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")

    def get_similar_drugs(self, drug_name: str, method="chemical", top_n=10):
        if not self._is_loaded:
            return None, "AI Service not loaded! Call load_data() first."

        search_term = drug_name.lower().strip()
        
        # 1. Try Direct Lookup
        drug_id = self.drug_ids.get(search_term)

        # 2. If not found, check manual aliases
        if not drug_id and search_term in self.manual_aliases:
            real_name = self.manual_aliases[search_term]
            drug_id = self.drug_ids.get(real_name)

        if not drug_id:
            return None, f"Drug '{drug_name}' not found."

        # 3. Choose Matrix and Math Logic
        if method == "chemical":
            matrix = self.chemical_matrix
            if drug_id not in matrix.index:
                return None, "Drug has no chemical features."
            
            target_vec = matrix.loc[drug_id].values.reshape(1, -1)
            # CHEMISTRY USES TANIMOTO (Intersection / Union)
            scores = SimilarityEngine.calculate_tanimoto(target_vec, matrix.values).flatten()
            
        elif method == "network":
            matrix = self.network_matrix
            if drug_id not in matrix.index:
                return None, "Drug has no biological network features."
            
            target_vec = matrix.loc[drug_id].values.reshape(1, -1)
            # BIOLOGY USES COSINE (RWR Embeddings)
            scores = RWR.calculate_cosine(target_vec, matrix.values).flatten()
            
        else:
            return None, "Invalid method"

        # 4. Format Results
        results = []
        for idx, score in sorted(zip(matrix.index, scores), key=lambda x: x[1], reverse=True)[:top_n+1]:
            if idx == drug_id: continue
            results.append({
                "name": self.drug_names.get(idx, "Unknown"),
                "score": round(float(score), 4) # Converted to float for clean output
            })
            
        return results, None