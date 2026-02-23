import pandas as pd
import random
from sqlalchemy import create_engine
from src.models.tier_1_similarity.advanced_fusion import AdvancedFusionModel
from src.evaluation.metrics import calculate_ddi_metrics, print_performance_comparison

DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"

def run_rigorous_evaluation():
    engine = create_engine(DB_URL)
    model = AdvancedFusionModel()
    
    # 1. Get Ground Truth Positives (Real Interactions)
    print("📥 Loading real interactions...")
    positives = pd.read_sql("""
        SELECT d1.id as id1, d1.smiles as smiles1, d2.id as id2, d2.smiles as smiles2
        FROM drug_interactions di
        JOIN drugs d1 ON di.drug_a_id = d1.id
        JOIN drugs d2 ON di.drug_b_id = d2.id
        LIMIT 500
    """, engine)

    # 2. Simple Negative Sampling (Fake Interactions)
    # Creating pairs that aren't in the interaction table
    print("🧪 Generating negative samples...")
    all_drug_ids = pd.read_sql("SELECT id, smiles FROM drugs WHERE smiles IS NOT NULL", engine)
    negatives = []
    while len(negatives) < len(positives):
        d1 = all_drug_ids.sample(1).iloc[0]
        d2 = all_drug_ids.sample(1).iloc[0]
        if d1['id'] != d2['id']:
            negatives.append({'id1': d1['id'], 'smiles1': d1['smiles'], 'id2': d2['id'], 'smiles2': d2['smiles']})
    
    neg_df = pd.DataFrame(negatives)

    # 3. Scoring
    y_true = [1] * len(positives) + [0] * len(neg_df)
    full_data = pd.concat([positives, neg_df])
    
    baseline_scores = []
    advanced_scores = []

    print("⚖️  Benchmarking models...")
    for _, row in full_data.iterrows():
        d1 = {'id': row['id1'], 'smiles': row['smiles1']}
        d2 = {'id': row['id2'], 'smiles': row['smiles2']}
        
        # Baseline = Chemical Only
        baseline_scores.append(model.get_chemical_similarity(d1['smiles'], d2['smiles']))
        
        # Advanced = Fused (Chemical + Phenotypic)
        advanced_scores.append(model.predict_fusion_score(d1, d2))

    # 4. Final Comparison
    m1 = calculate_ddi_metrics(y_true, baseline_scores)
    m2 = calculate_ddi_metrics(y_true, advanced_scores)
    
    print_performance_comparison(m1, m2)

if __name__ == "__main__":
    run_rigorous_evaluation()