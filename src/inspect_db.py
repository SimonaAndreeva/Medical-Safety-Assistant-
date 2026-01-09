import pandas as pd
from sqlalchemy import create_engine

# Connect
DB_URL = "postgresql://admin:12345@127.0.0.1:5432/medical_safety_db"
engine = create_engine(DB_URL)

def peek_table(table_name, limit=3):
    print(f"\n🔎 TABLE: {table_name.upper()}")
    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {limit}", engine)
    print(df.to_string(index=False))
    print("-" * 60)

print("📊 DATABASE INSPECTION REPORT")
print("=" * 60)

#Chemicals
peek_table("drugs")

#Safety (The Target for ML)
peek_table("drug_interactions")

#Biology (The Network)
peek_table("drug_targets")

#The New PPI Network (Protein-Protein)
peek_table("protein_interactions")
