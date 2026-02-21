import sys
import os
import numpy as np
import pandas as pd

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from src.config import settings
from src.models.tier_3_hin.hin_model import HINService

# 🏆 GOLD STANDARD
GOLD_STANDARD_FAMILIES = {
    "PDE5 Inhibitors": ["sildenafil", "tadalafil", "vardenafil", "udenafil"],
    "NSAIDs": ["ibuprofen", "naproxen", "diclofenac", "aspirin", "celecoxib"],
    "Statins": ["atorvastatin", "simvastatin", "rosuvastatin", "lovastatin"],
    "Beta Blockers": ["metoprolol", "atenolol", "propranolol", "bisoprolol"],
    "SSRIs": ["fluoxetine", "sertraline", "paroxetine", "citalopram"],
    "PPIs": ["omeprazole", "pantoprazole", "lansoprazole", "esomeprazole"],
    "Benzodiazepines": ["diazepam", "alprazolam", "lorazepam", "clonazepam"]
}

def calculate_metrics_for_query(ranked_results, true_positive_ids, k=10):
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    rr = 0.0 
    
    result_ids = [r['id'] for r in ranked_results]
    
    # 1. Calculate MRR and Hits@K
    first_correct_rank = float('inf')
    for i, res_id in enumerate(result_ids):
        rank = i + 1 
        if res_id in true_positive_ids:
            if rank < first_correct_rank:
                first_correct_rank = rank
            
            if rank <= 1: hits_1 = 1
            if rank <= 5: hits_5 = 1
            if rank <= 10: hits_10 = 1
            break 
            
    if first_correct_rank != float('inf'):
        rr = 1.0 / first_correct_rank

    # 2. Calculate Precision, Recall, and F1 @ K
    top_k_ids = set(result_ids[:k])
    tp_found = len(top_k_ids.intersection(true_positive_ids))
    total_relevant = len(true_positive_ids)
    
    precision_at_k = tp_found / k if k > 0 else 0.0
    recall_at_k = tp_found / total_relevant if total_relevant > 0 else 0.0
    
    f1_at_k = 0.0
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        
    return rr, hits_1, hits_5, hits_10, precision_at_k, recall_at_k, f1_at_k

def rigorous_evaluation():
    print(f"📊 STARTING SCIENTIFIC EVALUATION (F1 Included)")
    print(f"   Settings: r={settings.HIN_RWR_RESTART_PROB}")
    print("-" * 60)

    service = HINService()
    if not service.load_data():
        return

    global_mrr, global_h1, global_h5, global_h10 = [], [], [], []
    global_p10, global_r10, global_f1_10 = [], [], []
    
    for family, member_names in GOLD_STANDARD_FAMILIES.items():
        family_map = {}
        for name in member_names:
            did, _ = service._resolve_name_to_id(name)
            if did: family_map[name] = did

        valid_members = list(family_map.keys())
        if len(valid_members) < 2: continue

        for query_name in valid_members:
            query_id = family_map[query_name]
            true_positives = {fid for fname, fid in family_map.items() if fname != query_name}
            
            results, _ = service.get_similar_drugs(query_name, top_n=100)
            if not results: continue
                
            mrr, h1, h5, h10, p10, r10, f1_10 = calculate_metrics_for_query(results, true_positives, k=10)
            
            global_mrr.append(mrr)
            global_h1.append(h1)
            global_h5.append(h5)
            global_h10.append(h10)
            global_p10.append(p10)
            global_r10.append(r10)
            global_f1_10.append(f1_10)

    print("\n" + "=" * 60)
    print("🎓 FINAL THESIS METRICS (Leave-One-Out CV)")
    print("=" * 60)
    if len(global_mrr) > 0:
        print(f"🔹 MRR:               {np.mean(global_mrr):.4f}")
        print(f"🔹 Hits @ 1:          {np.mean(global_h1):.4f}")
        print(f"🔹 Hits @ 5:          {np.mean(global_h5):.4f}")
        print(f"🔹 Hits @ 10:         {np.mean(global_h10):.4f}")
        print("-" * 60)
        print(f"🔹 Precision @ 10:    {np.mean(global_p10):.4f}  (Max possible ~0.35 due to small family sizes)")
        print(f"🔹 Recall @ 10:       {np.mean(global_r10):.4f}")
        print(f"🔹 F1 Score @ 10:     {np.mean(global_f1_10):.4f}")
        print("=" * 60)

if __name__ == "__main__":
    rigorous_evaluation()