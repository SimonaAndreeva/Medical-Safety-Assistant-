import sys
import os

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

import pandas as pd
from sqlalchemy import create_engine, text
from src.config import settings

def export_network():
    print("📦 Exporting PPI Network to CSV...")
    
    engine = create_engine(settings.DB_URL)
    
    with engine.connect() as conn:
        # 1. Check available columns
        print("   -> Checking table structure...")
        columns_query = text("SELECT * FROM protein_interactions LIMIT 1")
        columns = pd.read_sql_query(columns_query, conn).columns.tolist()
        print(f"      Columns found: {columns}")
        
        # Determine the 'score' column
        score_col = "1.0" # Default value if no score exists
        if 'combined_score' in columns:
            score_col = "combined_score"
        elif 'score' in columns:
            score_col = "score"
        elif 'confidence' in columns:
            score_col = "confidence"
            
        print(f"      Using '{score_col}' as edge weight.")

        # 2. Read the Table
        # We construct the query dynamically based on what we found
        if score_col == "1.0":
            query = text(f"""
                SELECT 
                    protein_a_uniprot AS source, 
                    protein_b_uniprot AS target,
                    1.0 AS weight
                FROM protein_interactions
            """)
        else:
            query = text(f"""
                SELECT 
                    protein_a_uniprot AS source, 
                    protein_b_uniprot AS target,
                    {score_col} AS weight
                FROM protein_interactions
            """)
        
        print("   -> Reading from database (this might take a moment)...")
        df = pd.read_sql_query(query, conn)
    
    # 3. Save to CSV
    output_file = os.path.join(project_root, "data/processed/full_ppi_network.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    print(f"✅ Success! Exported {len(df)} interactions.")
    print(f"📂 File saved at: {output_file}")

if __name__ == "__main__":
    export_network()