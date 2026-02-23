import pandas as pd
from sqlalchemy import create_engine
from src.models.tier_1_similarity.advanced_fusion import AdvancedFusionModel

# Database connection
DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"

def test_modality_fusion():
    engine = create_engine(DB_URL)
    model = AdvancedFusionModel()

    # 1. Fetch a sample of drugs that have both SMILES and PubChem CIDs
    # 1. Fetch a sample. We'll use COALESCE or just omit name if it fails
    query = """
        SELECT id, smiles 
        FROM drugs 
        WHERE smiles IS NOT NULL 
        AND pubchem_cid IS NOT NULL 
        LIMIT 5
    """
    sample_drugs = pd.read_sql(query, engine)
    
    if len(sample_drugs) < 2:
        print("❌ Not enough drugs with complete data to test.")
        return

    print(f"\n🧪 Testing Fusion for {len(sample_drugs)} drugs...")
    print("-" * 60)

    # 2. Compare pairs
    results = []
    for i in range(len(sample_drugs)):
        for j in range(i + 1, len(sample_drugs)):
            d1 = sample_drugs.iloc[i]
            d2 = sample_drugs.iloc[j]
            
            # Calculate individual and fused scores
            s_chem = model.get_chemical_similarity(d1['smiles'], d2['smiles'])
            s_pheno = model.get_phenotypic_similarity(d1['id'], d2['id'])
            fused = model.predict_fusion_score(d1, d2)
            
            results.append({
                "Drug 1 ID": d1['id'],
                "Drug 2 ID": d2['id'],
                "Chem Sim": round(s_chem, 4),
                "Pheno Sim": round(s_pheno, 4),
                "Fused Score": round(fused, 4)
            })

    # 3. Display Results
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # 4. Success Check
    if df_results['Pheno Sim'].sum() > 0:
        print("\n✅ SUCCESS: Phenotypic data is being successfully fused with Chemical data.")
    else:
        print("\n⚠️  WARNING: Phenotypic scores are all 0. This means these specific drugs aren't in SIDER.")

if __name__ == "__main__":
    test_modality_fusion()