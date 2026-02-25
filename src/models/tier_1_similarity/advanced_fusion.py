import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys

# --- PATHS & CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import settings
PROCESSED_DATA_DIR = "data/processed/"
PHENO_MATRIX_PATH = os.path.join(PROCESSED_DATA_DIR, "phenotypic_matrix.pkl")

class AdvancedFusionModel:
    def __init__(self):
        # Load the side-effect matrix we just built
        with open(PHENO_MATRIX_PATH, 'rb') as f:
            self.pheno_matrix = pickle.load(f) # Rows=DrugIDs, Cols=SideEffects
        
        self.known_pheno_ids = set(self.pheno_matrix.index)
        self._fp_cache = {}  # Upgrade 2: Fingerprint Cache
        print(f"🧬 Fusion Model Initialized with {len(self.known_pheno_ids)} phenotypic drug profiles.")

    def _get_fingerprint(self, smiles):
        """Helper to compute and cache Morgan fingerprints safely."""
        if not smiles or not isinstance(smiles, str):
            return None
        
        if smiles in self._fp_cache:
            return self._fp_cache[smiles]
            
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            self._fp_cache[smiles] = None
            return None
            
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        self._fp_cache[smiles] = fp
        return fp

    def get_chemical_similarity(self, smiles1, smiles2):
        """Calculates Tanimoto similarity for 2D structures."""
        fp1 = self._get_fingerprint(smiles1)
        fp2 = self._get_fingerprint(smiles2)
        
        if not fp1 or not fp2: 
            return 0.0
            
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def get_phenotypic_similarity(self, id1, id2):
        """Calculates Jaccard similarity for side-effect overlap."""
        if id1 not in self.known_pheno_ids or id2 not in self.known_pheno_ids:
            return 0.0 # Return 0 if we lack SIDER data for these drugs
            
        vec1 = self.pheno_matrix.loc[id1].values.reshape(1, -1)
        vec2 = self.pheno_matrix.loc[id2].values.reshape(1, -1)
        
        # Jaccard = Intersection / Union
        intersection = np.logical_and(vec1, vec2).sum()
        union = np.logical_or(vec1, vec2).sum()
        return intersection / union if union > 0 else 0.0

    def predict_fusion_score(self, drug1, drug2, weight_pheno=None):
        """
        Fuses modalities. Industry standard: weight phenotypic 
        higher if available, as it's closer to clinical reality.
        """
        s_chem = self.get_chemical_similarity(drug1['smiles'], drug2['smiles'])
        s_pheno = self.get_phenotypic_similarity(drug1['id'], drug2['id'])
        
        # If no side effect data, rely entirely on chemical
        if s_pheno == 0:
            return s_chem
            
        # Use config driven weight, fallback to 0.6 if undefined
        if weight_pheno is None:
            weight_pheno = getattr(settings, 'FUSION_WEIGHT_PHENO', 0.6)
            
        # Weighted Fusion (HNAI approach)
        return (weight_pheno * s_pheno) + ((1 - weight_pheno) * s_chem)

# Example Usage logic
if __name__ == "__main__":
    # This would normally be fed from your database
    drug_a = {'id': 1, 'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'} # Aspirin
    drug_b = {'id': 2, 'smiles': 'CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC'} # Cocaine
    
    model = AdvancedFusionModel()
    score = model.predict_fusion_score(drug_a, drug_b)
    print(f"🔥 Combined Fusion Similarity Score: {score:.4f}")