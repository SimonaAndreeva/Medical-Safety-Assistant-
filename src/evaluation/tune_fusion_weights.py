import pandas as pd
import numpy as np
import os
import sys

# --- PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from sqlalchemy import create_engine
from src.config import settings
from src.models.tier_1_similarity.advanced_fusion import AdvancedFusionModel
from src.evaluation.metrics import calculate_ddi_metrics

def generate_evaluation_dataset(engine, sample_size=500):
    """
    Fetch raw interactions and generate negative samples.
    """
    print(f"📥 Loading {sample_size} real interactions...")
    positives = pd.read_sql(f"""
        SELECT d1.id as id1, d1.smiles as smiles1, d2.id as id2, d2.smiles as smiles2
        FROM drug_interactions di
        JOIN drugs d1 ON di.drug_a_id = d1.id
        JOIN drugs d2 ON di.drug_b_id = d2.id
        LIMIT {sample_size}
    """, engine)

    print(f"🧪 Generating {len(positives)} negative samples...")
    all_drug_ids = pd.read_sql("SELECT id, smiles FROM drugs WHERE smiles IS NOT NULL", engine)
    negatives = []
    while len(negatives) < len(positives):
        d1 = all_drug_ids.sample(1).iloc[0]
        d2 = all_drug_ids.sample(1).iloc[0]
        if d1['id'] != d2['id']:
            negatives.append({'id1': d1['id'], 'smiles1': d1['smiles'], 'id2': d2['id'], 'smiles2': d2['smiles']})
    
    neg_df = pd.DataFrame(negatives)
    
    y_true = [1] * len(positives) + [0] * len(neg_df)
    full_data = pd.concat([positives, neg_df])
    return full_data, np.array(y_true)

def tune_fusion_weights():
    engine = create_engine(settings.DB_URL)
    model = AdvancedFusionModel()
    
    full_data, y_true = generate_evaluation_dataset(engine, sample_size=500)
    
    print("\n🔍 Precomputing modality scores (Vectorized Processing)...")
    smiles_A = full_data['smiles1'].tolist()
    smiles_B = full_data['smiles2'].tolist()
    ids_A = full_data['id1'].tolist()
    ids_B = full_data['id2'].tolist()
    
    import time
    start = time.perf_counter()
    s_chem_arr = model.get_chemical_similarity(smiles_A, smiles_B)
    s_pheno_arr = model.get_phenotypic_similarity(ids_A, ids_B)
    end = time.perf_counter()
    print(f"✅ Fast Matrix math completed in {(end - start):.4f} seconds!")
    
    weights = np.arange(0.0, 1.1, 0.1)
    
    print(f"\n⚖️  Grid Searching Phenotypic Weights ({len(weights)} steps)...")
    print("-" * 50)
    print(f"{'Weight (Pheno)':<15} | {'AUPR':<10} | {'AUROC':<10}")
    print("-" * 50)
    
    best_aupr = -1
    best_weight = 0.0
    
    for w in weights:
        # Vectorized implementation of the model's logic
        # If s_pheno == 0, rely on s_chem
        final_scores = np.where(s_pheno_arr == 0, s_chem_arr, (w * s_pheno_arr) + ((1 - w) * s_chem_arr))
        
        metrics = calculate_ddi_metrics(y_true, final_scores)
        aupr = metrics['AUPR']
        auroc = metrics['AUROC']
        
        print(f"{w:<15.1f} | {aupr:<10.4f} | {auroc:<10.4f}")
        
        if aupr > best_aupr:
            best_aupr = aupr
            best_weight = w
            
    print("-" * 50)
    print(f"🎯 OPTIMAL WEIGHT FOUND: {best_weight:.1f} (AUPR: {best_aupr:.4f})")
    print("Action item: Update settings.FUSION_WEIGHT_PHENO in src.config.py")

if __name__ == "__main__":
    tune_fusion_weights()
