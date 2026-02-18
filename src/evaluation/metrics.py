import sys
import os
import numpy as np
import pandas as pd

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from src.web.services import DrugSimilarityService
from src.config import settings

# 🏆 THE GOLD STANDARD BENCHMARK 🏆
# These are undeniable medical facts. Drugs in the same list DO the same thing.
# Your AI *should* find these connections if the RWR is working.
GOLD_STANDARD_FAMILIES = {
    "PDE5 Inhibitors (ED)": ["sildenafil", "tadalafil", "vardenafil", "udenafil"],
    "NSAIDs (Pain)": ["ibuprofen", "naproxen", "diclofenac", "aspirin", "celecoxib"],
    "Statins (Cholesterol)": ["atorvastatin", "simvastatin", "rosuvastatin", "lovastatin"],
    "Beta Blockers (Heart)": ["metoprolol", "atenolol", "propranolol", "bisoprolol"],
    "SSRIs (Antidepressants)": ["fluoxetine", "sertraline", "paroxetine", "citalopram"],
    "PPIs (Stomach Acid)": ["omeprazole", "pantoprazole", "lansoprazole", "esomeprazole"],
    "Benzodiazepines (Anxiety)": ["diazepam", "alprazolam", "lorazepam", "clonazepam"]
}

def evaluate_model(top_n=20):
    print(f"🔬 STARTING RWR HYPERPARAMETER EVALUATION")
    print(f"   Settings: r={settings.RWR_RESTART_PROB}, iter={settings.RWR_MAX_ITER}")
    print("-" * 60)

    # 1. Load the AI Brain
    service = DrugSimilarityService()
    service.load_data()
    
    if service.network_matrix is None:
        print("❌ Error: Network models not found. Run 'build_network.py' first.")
        return

    total_score = 0
    total_tests = 0
    
    # 2. Run the Exam
    for family_name, members in GOLD_STANDARD_FAMILIES.items():
        print(f"\n📂 Testing Family: {family_name}")
        family_hits = 0
        family_tests = 0
        
        # Test every drug in the family against the others
        for drug in members:
            # Skip if drug isn't in our DB
            if drug not in service.drug_ids:
                continue
                
            # Ask AI for Top N Biological Cousins
            results, err = service.get_similar_drugs(drug, method="network", top_n=top_n)
            
            if err or not results:
                continue

            # Extract the names of the suggested cousins
            suggested_names = [r['name'].lower() for r in results]
            
            # CHECK: Did it find the other family members?
            # (We remove the drug itself from the target list)
            targets = [m for m in members if m != drug]
            
            hits = 0
            for target in targets:
                if target in suggested_names:
                    hits += 1
            
            # Calculate Recall (How many of the targets did we find?)
            if len(targets) > 0:
                recall = hits / len(targets)
                family_hits += recall
                family_tests += 1
                
                # Visual Check for the User
                status = "✅" if recall > 0.5 else "⚠️"
                print(f"   {status} {drug:<15} found {hits}/{len(targets)} family members.")

        if family_tests > 0:
            avg_family_score = (family_hits / family_tests) * 100
            print(f"   >>> Family Accuracy: {avg_family_score:.1f}%")
            total_score += avg_family_score
            total_tests += 1

    # 3. Final Report Card
    if total_tests > 0:
        final_accuracy = total_score / total_tests
        print("\n" + "=" * 60)
        print(f"🎓 FINAL MODEL GRADE (Average Recall @ {top_n})")
        print(f"📊 ACCURACY: {final_accuracy:.2f}%")
        print("=" * 60)
        
        # Interpretation Guide
        if final_accuracy > 80:
            print("🌟 EXCELLENT. The RWR is perfectly tuned.")
        elif final_accuracy > 60:
            print("✅ GOOD. The model is solid but could be tighter.")
        else:
            print("⚠️ NEEDS TUNING. Try changing 'RWR_RESTART_PROB' in config.py.")
    else:
        print("❌ No tests could be run (check if drugs exist in DB).")

if __name__ == "__main__":
    evaluate_model()