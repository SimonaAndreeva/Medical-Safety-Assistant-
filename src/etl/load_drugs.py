import pandas as pd
from sqlalchemy import create_engine

# --- CONFIGURATION ---
# Source: DrugCentral (Port 5433)
# We use 127.0.0.1 to avoid Windows "localhost" IPv6 issues
SRC_DB_URL = "postgresql://postgres:rawpassword@127.0.0.1:5433/drugcentral"

# Target: Medical Safety DB (Port 5432)
# Updated password to match your environment
DST_DB_URL = "postgresql://admin:12345@127.0.0.1:5432/medical_safety_db"

def load_data():
    print("🔌 Connecting to databases...")
    src_engine = create_engine(SRC_DB_URL)
    dst_engine = create_engine(DST_DB_URL)

    # --- STEP 1: LOAD DRUGS ---
    print("\n📦 STEP 1: Loading Drugs...")
    query_drugs = """
    SELECT 
        id::text AS external_id,
        name     AS generic_name,
        smiles,
        inchikey,
        'DrugCentral' AS source
    FROM structures
    WHERE smiles IS NOT NULL AND inchikey IS NOT NULL
    """
    df_drugs = pd.read_sql(query_drugs, src_engine)
    
    # Remove duplicates based on InChIKey (Chemical ID)
    df_drugs = df_drugs.drop_duplicates(subset=['inchikey'])
    
    # Save to DB
    df_drugs.to_sql('drugs', dst_engine, if_exists='append', index=False, method='multi', chunksize=500)
    print(f"✅ Loaded {len(df_drugs)} unique drugs.")


    # --- HELPER: Get Mapping (Generic Name -> Our Internal ID) ---
    # CORRECTION: We map by NAME because the interaction table uses names, not IDs.
    print("   -> Building Name map...")
    map_query = "SELECT generic_name, id FROM drugs"
    name_map = pd.read_sql(map_query, dst_engine)
    
    # Create dictionary: {'Ibuprofen': 1, 'Aspirin': 2}
    # We uppercase the keys to ensure matching works even if case differs
    name_dict = dict(zip(name_map.generic_name.str.upper(), name_map.id))

    # We also keep the ID map for Synonyms/ATC which DO use IDs
    id_map_query = "SELECT external_id, id FROM drugs"
    id_df = pd.read_sql(id_map_query, dst_engine)
    id_dict = dict(zip(id_df.external_id, id_df.id))


    # --- STEP 2: LOAD SYNONYMS (Uses ID Mapping) ---
    print("\n📦 STEP 2: Loading Synonyms...")
    query_syn = "SELECT id::text as struct_id, name as synonym_name FROM synonyms"
    df_syn = pd.read_sql(query_syn, src_engine)
    df_syn['drug_id'] = df_syn['struct_id'].map(id_dict)
    df_syn = df_syn.dropna(subset=['drug_id'])
    df_syn[['drug_id', 'synonym_name']].to_sql('drug_synonyms', dst_engine, if_exists='append', index=False, method='multi', chunksize=1000)
    print(f"✅ Loaded {len(df_syn)} synonyms.")


    # --- STEP 3: LOAD ATC CODES (Uses ID Mapping) ---
    print("\n📦 STEP 3: Loading ATC Codes...")
    query_atc = "SELECT struct_id::text, atc_code FROM struct2atc"
    df_atc = pd.read_sql(query_atc, src_engine)
    df_atc['drug_id'] = df_atc['struct_id'].map(id_dict)
    df_atc = df_atc.dropna(subset=['drug_id'])
    df_atc[['drug_id', 'atc_code']].to_sql('drug_atc_codes', dst_engine, if_exists='append', index=False, method='multi', chunksize=1000)
    print(f"✅ Loaded {len(df_atc)} ATC codes.")


    # --- STEP 4: LOAD INTERACTIONS (Uses NAME Mapping) ---
    print("\n📦 STEP 4: Loading Interactions...")
    query_ddi = """
    SELECT 
        drug_class1 as name_a,
        drug_class2 as name_b,
        description,
        ddi_risk,
        source_id
    FROM ddi
    """
    df_ddi = pd.read_sql(query_ddi, src_engine)
    
    # Map using the NAME dictionary (Upper case to match)
    df_ddi['drug_a_id'] = df_ddi['name_a'].str.upper().map(name_dict)
    df_ddi['drug_b_id'] = df_ddi['name_b'].str.upper().map(name_dict)
    
    # Keep only rows where we found BOTH drugs
    df_ddi = df_ddi.dropna(subset=['drug_a_id', 'drug_b_id'])
    
    df_ddi = df_ddi.rename(columns={'ddi_risk': 'risk_level'})
    
    df_ddi[['drug_a_id', 'drug_b_id', 'description', 'risk_level', 'source_id']].to_sql(
        'drug_interactions', 
        dst_engine, 
        if_exists='append', 
        index=False, 
        method='multi', 
        chunksize=500
    )
    print(f"✅ Loaded {len(df_ddi)} interactions.")
    
    print("\n🎉 ALL DONE! Your database is now populated with rich data.")

if __name__ == "__main__":
    load_data()