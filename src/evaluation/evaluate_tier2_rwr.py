import pandas as pd
import numpy as np
import random
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.tier_2_network.ppi_network_model import PPINetworkModel

DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"

def evaluate_rwr_model():
    print("🚀 Initializing Tier II RWR Evaluation...")
    engine = create_engine(DB_URL)
    model = PPINetworkModel(db_url=DB_URL)

    # 1. Fetch POSITIVE samples (Real Interactions)
    print("📥 Fetching Ground Truth Interactions...")
    pos_query = """
        SELECT DISTINCT di.drug_a_id, di.drug_b_id
        FROM drug_interactions di
        INNER JOIN drug_targets dt1 ON di.drug_a_id = dt1.drug_id
        INNER JOIN drug_targets dt2 ON di.drug_b_id = dt2.drug_id
    """
    positives = pd.read_sql(pos_query, engine).values.tolist()
    
    # Sample 500 positive interactions to keep the test reasonably fast
    if len(positives) > 500:
        positives = random.sample(positives, 500)

    # 2. Fetch NEGATIVE samples (Random drug pairs with targets, assumed non-interacting)
    print("📥 Generating Negative Samples...")
    drugs_with_targets = pd.read_sql("SELECT DISTINCT drug_id FROM drug_targets", engine)['drug_id'].tolist()
    
    negatives = []
    pos_set = set(tuple(sorted(pair)) for pair in positives)
    
    while len(negatives) < len(positives):
        d1, d2 = random.sample(drugs_with_targets, 2)
        pair = tuple(sorted([d1, d2]))
        if pair not in pos_set and pair not in negatives:
            negatives.append(pair)

    # 3. Combine and Label Data
    test_pairs = positives + negatives
    y_true = [1] * len(positives) + [0] * len(negatives)
    y_scores = []

    # 4. Run the Evaluation
    print(f"🧪 Testing {len(test_pairs)} pairs through the RWR Network...")
    for i, (d1, d2) in enumerate(test_pairs):
        if i % 100 == 0 and i > 0:
            print(f"   Processed {i}/{len(test_pairs)} pairs...")
        
        score = model.get_network_similarity(d1, d2)
        y_scores.append(score)

    # 5. Calculate Metrics Directly
    print("\n📊 TIER II RWR STANDALONE RESULTS:")
    print("-" * 35)
    
    # Handle edge case where all scores might be 0 due to sparse data
    if sum(y_scores) == 0:
        print("⚠️ Warning: All similarity scores were 0.0. Network is too sparse to calculate metrics.")
    else:
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        print(f"AUROC:           {auroc:.4f}")
        print(f"AUPR:            {aupr:.4f}")
        print(f"Avg Precision:   {aupr:.4f}")

if __name__ == "__main__":
    evaluate_rwr_model()