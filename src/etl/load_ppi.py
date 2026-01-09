import pandas as pd
import requests
import io
from sqlalchemy import create_engine

# --- CONFIGURATION ---
DST_DB_URL = "postgresql://admin:12345@127.0.0.1:5432/medical_safety_db"
OMNIPATH_URL = "https://omnipathdb.org/interactions"

def load_ppi():
    print("Connecting to database...")
    dst_engine = create_engine(DST_DB_URL)

    # Get Proteins
    print("Fetching unique proteins from database...")
    query_proteins = "SELECT DISTINCT target_uniprot_id FROM drug_targets"
    df_proteins = pd.read_sql(query_proteins, dst_engine)
    
    unique_proteins = set(df_proteins['target_uniprot_id'].dropna().tolist())
    print(f"   -> Found {len(unique_proteins)} unique protein nodes.")

    if not unique_proteins:
        print("No proteins found! Run load_targets.py first.")
        return

    # Query OmniPath
    print("Querying OmniPath for human interactome...")
    
    # FIXED PARAMS: Removed 'organism' (default is human) and complex 'fields'
    params = {
        'format': 'tab',
        'genesymbols': 'no',
    }
    
    try:
        response = requests.get(OMNIPATH_URL, params=params, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}")
        return

    print("   -> Parsing network data...")
    df_ppi = pd.read_csv(io.StringIO(response.text), sep='\t')
    
    # Clean column names
    df_ppi.columns = df_ppi.columns.str.strip()
    
    # Check if the data is valid
    if 'source' not in df_ppi.columns or 'target' not in df_ppi.columns:
        print(f"ERROR: Unexpected columns: {df_ppi.columns.tolist()}")
        print("   -> API Response Start:", response.text[:200])
        return

    # Filter
    print("Filtering for relevant interactions...")
    
    df_ppi = df_ppi.rename(columns={
        'source': 'protein_a_uniprot',
        'target': 'protein_b_uniprot'
    })
    
    # Keep only interactions where BOTH proteins are in the set
    relevant_mask = (
        df_ppi['protein_a_uniprot'].isin(unique_proteins) & 
        df_ppi['protein_b_uniprot'].isin(unique_proteins)
    )
    df_filtered = df_ppi[relevant_mask].copy()
    
    # --- FILL MISSING DATA ---
    # Since we removed the 'fields' parameter, these columns might not exist.
    # We fill them with defaults to satisfy our database schema.
    
    df_filtered['source'] = 'OmniPath'
    
    if 'is_directed' in df_filtered.columns:
        df_filtered['is_directed'] = df_filtered['is_directed'].astype(bool)
    else:
        df_filtered['is_directed'] = False 
        
    if 'consensus_score' in df_filtered.columns:
        df_filtered['consensus_score'] = df_filtered['consensus_score'].fillna(0)
    else:
        # If no score, we use the number of references/sources as a proxy if available, 
        # otherwise just 1 (it exists).
        if 'n_references' in df_filtered.columns:
             df_filtered['consensus_score'] = df_filtered['n_references']
        else:
             df_filtered['consensus_score'] = 1

    # Remove duplicates
    df_filtered = df_filtered.drop_duplicates(subset=['protein_a_uniprot', 'protein_b_uniprot'])

    print(f"   -> Found {len(df_filtered)} interactions between drug targets.")

    # Save
    if not df_filtered.empty:
        print("Saving to 'protein_interactions' table...")
        df_filtered[['protein_a_uniprot', 'protein_b_uniprot', 'source', 'is_directed', 'consensus_score']].to_sql(
            'protein_interactions', 
            dst_engine, 
            if_exists='append', 
            index=False, 
            method='multi', 
            chunksize=1000
        )
        print("Success! PPI Network loaded.")
    else:
        print("No interactions found.")

if __name__ == "__main__":
    load_ppi()