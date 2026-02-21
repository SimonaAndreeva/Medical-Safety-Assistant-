import pandas as pd
from sqlalchemy import create_engine, text

# ==========================================
# ⚙️ CONFIGURATION 
# ==========================================
# Source: The Raw DrugCentral DB
SRC_DB_URL = "postgresql://postgres:rawpassword@127.0.0.1:5433/drugcentral"

# Target: Medical Safety DB 
DST_DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"
# ==========================================

def migrate_pubchem_cids():
    print("🔌 Connecting to databases...")
    dc_engine = create_engine(SRC_DB_URL)
    safety_engine = create_engine(DST_DB_URL)

    # 1. Fetch the Mapping Table from DrugCentral
    print("📥 Extracting PubChem mappings from DrugCentral...")
    query = """
        SELECT struct_id AS external_id, identifier AS pubchem_cid 
        FROM identifier 
        WHERE id_type = 'PUBCHEM_CID'
    """
    try:
        mapping_df = pd.read_sql(query, dc_engine)
        print(f"✅ Found {len(mapping_df)} PubChem CIDs in DrugCentral.")
    except Exception as e:
        print(f"❌ Failed to read from DrugCentral: {e}")
        return

    mapping_df['external_id'] = mapping_df['external_id'].astype(str)
    mapping_df = mapping_df.drop_duplicates(subset=['external_id'])

    # 2. Update the Safety Database
    print("📤 Updating main Safety Database...")
    try:
        safety_drugs_df = pd.read_sql("SELECT external_id FROM drugs", safety_engine)
        existing_ids = set(safety_drugs_df['external_id'].astype(str))
        print(f"🔍 Found {len(existing_ids)} drugs currently in your safety_db.")
    except Exception as e:
        print(f"❌ Failed to read from Safety DB. Is it running? Error: {e}")
        return

    mapping_df = mapping_df[mapping_df['external_id'].isin(existing_ids)]
    print(f"🎯 Ready to update {len(mapping_df)} matching drugs.")

    with safety_engine.connect() as conn:
        conn.execute(text("ALTER TABLE drugs ADD COLUMN IF NOT EXISTS pubchem_cid TEXT;"))
        conn.commit()

        if not mapping_df.empty:
            mapping_df.to_sql('temp_pubchem_mapping', conn, index=False, if_exists='replace')
            
            update_query = text("""
                UPDATE drugs 
                SET pubchem_cid = temp_pubchem_mapping.pubchem_cid 
                FROM temp_pubchem_mapping 
                WHERE drugs.external_id = temp_pubchem_mapping.external_id;
            """)
            result = conn.execute(update_query)
            conn.commit()
            
            conn.execute(text("DROP TABLE temp_pubchem_mapping;"))
            conn.commit()
            
            print(f"🎉 Successfully updated {result.rowcount} drugs with their PubChem CIDs!")
        else:
            print("⚠️ No matching drugs found to update.")

if __name__ == "__main__":
    migrate_pubchem_cids()