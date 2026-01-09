import pandas as pd
from sqlalchemy import create_engine

# --- CONFIGURATION ---
# Source: DrugCentral (Port 5433)
SRC_DB_URL = "postgresql://postgres:rawpassword@127.0.0.1:5433/drugcentral"

# Target: Medical Safety DB (Port 5432)
DST_DB_URL = "postgresql://admin:12345@127.0.0.1:5432/medical_safety_db"

def load_targets():
    print("Connecting to databases...")
    src_engine = create_engine(SRC_DB_URL)
    dst_engine = create_engine(DST_DB_URL)

    print("\nBuilding Drug ID Map...")
    # We need to know which internal ID belongs to which DrugCentral ID
    # so we can link the targets correctly.
    map_query = "SELECT external_id, id FROM drugs"
    id_map = pd.read_sql(map_query, dst_engine)
    
    # Create dictionary: {'276': 5, '277': 6...}
    # This maps "DrugCentral ID" -> "Medical Safety DB ID"
    id_dict = dict(zip(id_map.external_id, id_map.id))
    print(f"   -> Found {len(id_dict)} drugs in your database.")

    print("\nLoading Drug Targets (Bioactivity)...")
    
    # Query to join activity table with target details
    query_targets = """
    SELECT 
        a.struct_id::text,
        t.accession as target_uniprot_id,
        t.name as target_name,
        a.act_type as action_type
    FROM act_table_full a
    JOIN target_component t ON a.target_id = t.id
    WHERE t.accession IS NOT NULL
    """
    
    # Load raw data from Source
    df_targets = pd.read_sql(query_targets, src_engine)
    print(f"   -> Extracted {len(df_targets)} raw interactions.")
    
    # Map 'struct_id' to internal 'drug_id'
    df_targets['drug_id'] = df_targets['struct_id'].map(id_dict)
    
    # Drop rows where the drug doesn't exist in clean DB
    df_targets = df_targets.dropna(subset=['drug_id'])
    
    # Remove duplicates
    df_targets = df_targets.drop_duplicates(subset=['drug_id', 'target_uniprot_id'])
    
    print(f"   -> Saving {len(df_targets)} verified target interactions...")

    # Save to 'drug_targets' table
    df_targets[['drug_id', 'target_name', 'target_uniprot_id', 'action_type']].to_sql(
        'drug_targets', 
        dst_engine, 
        if_exists='append', 
        index=False, 
        method='multi', 
        chunksize=1000
    )
    print("Success! Targets loaded.")

if __name__ == "__main__":
    load_targets()