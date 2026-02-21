import pandas as pd
from sqlalchemy import create_engine, text

# Use your verified credentials
DST_DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"

def run_audit():
    engine = create_engine(DST_DB_URL)
    
    queries = {
        "Total Drugs": "SELECT COUNT(*) FROM drugs",
        "Drugs with SMILES": "SELECT COUNT(*) FROM drugs WHERE smiles IS NOT NULL AND smiles != ''",
        "Drugs with PubChem CIDs": "SELECT COUNT(*) FROM drugs WHERE pubchem_cid IS NOT NULL",
        "Drugs with ATC Codes": "SELECT COUNT(DISTINCT drug_id) FROM drug_atc_codes",
        "Drugs with Targets": "SELECT COUNT(DISTINCT drug_id) FROM drug_targets",
        "Total Interactions (Edges)": "SELECT COUNT(*) FROM drug_interactions",
        "Total PPI Connections": "SELECT COUNT(*) FROM protein_interactions"
    }
    
    print("📊 DATASET HEALTH AUDIT")
    print("-" * 30)
    
    results = {}
    with engine.connect() as conn:
        for label, query in queries.items():
            count = conn.execute(text(query)).scalar()
            results[label] = count
            print(f"{label:25}: {count}")

    # Calculate Coverage Percentages
    total = results["Total Drugs"]
    if total > 0:
        print("\n📈 COVERAGE ANALYSIS")
        print(f"Chemical Coverage (SMILES): {(results['Drugs with SMILES']/total)*100:.2f}%")
        print(f"Phenotypic Key (PubChem):   {(results['Drugs with PubChem CIDs']/total)*100:.2f}%")
        print(f"Therapeutic (ATC):          {(results['Drugs with ATC Codes']/total)*100:.2f}%")
        print(f"Genomic (Targets):          {(results['Drugs with Targets']/total)*100:.2f}%")

if __name__ == "__main__":
    run_audit()