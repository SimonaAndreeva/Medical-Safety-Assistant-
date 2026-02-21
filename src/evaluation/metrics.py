import sys
import os
import numpy as np
import pandas as pd

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from src.config import settings
# We now use the HIN Service (The "Super Brain")
from src.web.hin_service import HINService

# 🏆 THE GOLD STANDARD BENCHMARK 🏆
# These are undeniable medical facts. Drugs in the same list DO the same thing.
# Your AI *should* find these connections if the HIN is working.
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
    print(f"🔬 STARTING HETEROGENEOUS NETWORK (HIN) EVALUATION")
    print(f"   Settings: r={settings.HIN_RWR_RESTART_PROB} (Restart Prob)")
    print("-" * 60)

    # 1. Load the AI Brain (HIN Service)
    service = HINService()
    success = service.load_data()
    
    if not success:
        print("❌ Error: HIN models not found. Run 'src/pipelines/build_hin_network.py' first.")
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
            # Ask AI for Top N Cousins
            # Note: HINService doesn't need 'method' arg, it only does Network
            results, err = service.get_similar_drugs(drug, top_n=top_n)
            
            # If drug not found or error, skip
            if err or not results:
                # Optional: Print why it failed (e.g. "Drug not found")
                # print(f"   [Skipped {drug}]: {err}") 
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
        print(f"🎓 FINAL HIN MODEL GRADE (Average Recall @ {top_n})")
        print(f"📊 ACCURACY: {final_accuracy:.2f}%")
        print("=" * 60)
        
        # Interpretation Guide
        if final_accuracy > 80:
            print("🌟 EXCELLENT. This is Thesis Quality.")
        elif final_accuracy > 60:
            print("✅ GOOD. Much better than random guessing.")
        else:
            print("⚠️ NEEDS TUNING. Check data connections.")
    else:
        print("❌ No tests could be run (check if drugs exist in DB).")

if __name__ == "__main__":
    evaluate_model()