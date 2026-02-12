import sys
import os

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

import pandas as pd
import networkx as nx
from pyvis.network import Network
from sqlalchemy import create_engine, text
from src.config import settings

def draw_system_map(drug_names):
    print(f"🗺️  Generating Multi-Relational Map for: {drug_names}...")
    
    engine = create_engine(settings.DB_URL)
    
    with engine.connect() as conn:
        # 1. Get Drug IDs
        # We format the list for SQL (e.g., 'aspirin','ibuprofen')
        formatted_names = tuple([n.lower() for n in drug_names])
        if len(formatted_names) == 1: formatted_names = f"('{formatted_names[0]}')"
        
        query_drugs = text(f"SELECT id, generic_name FROM drugs WHERE generic_name IN {formatted_names}")
        df_drugs = pd.read_sql_query(query_drugs, conn)
        
        if df_drugs.empty:
            print("❌ No drugs found.")
            return

        drug_ids = tuple(df_drugs['id'].astype(int).tolist()) # Convert to Python ints
        if len(drug_ids) == 1: drug_ids = f"({drug_ids[0]})"
        
        print(f"   -> Found {len(df_drugs)} drugs.")

        # 2. Get Drug-Target Relations (Edge Type 1)
        query_targets = text(f"""
            SELECT d.generic_name, dt.target_uniprot_id 
            FROM drug_targets dt
            JOIN drugs d ON dt.drug_id = d.id
            WHERE dt.drug_id IN {drug_ids}
        """)
        df_dt = pd.read_sql_query(query_targets, conn)
        targets = tuple(df_dt['target_uniprot_id'].unique())
        
        print(f"   -> Found {len(df_dt)} Drug-Target connections.")

        # 3. Get Protein-Protein Relations (Edge Type 2)
        if not targets:
            print("⚠️ No targets found. Graph will be just drugs.")
            df_ppi = pd.DataFrame()
        else:
            if len(targets) == 1: targets = f"('{targets[0]}')"
            
            # Limit PPIs to avoid explosion
            query_ppi = text(f"""
                SELECT protein_a_uniprot, protein_b_uniprot 
                FROM protein_interactions 
                WHERE protein_a_uniprot IN {targets} 
                   OR protein_b_uniprot IN {targets}
                LIMIT 1000
            """)
            df_ppi = pd.read_sql_query(query_ppi, conn)
            print(f"   -> Found {len(df_ppi)} Protein-Protein connections.")

    # 4. Build the Multi-Relational Graph
    G = nx.Graph()

    # --- LAYER 1: Drugs (Squares) ---
    for _, row in df_drugs.iterrows():
        G.add_node(row['generic_name'], 
                   label=row['generic_name'], 
                   title="Drug Node", 
                   color="#ff4444",  # Red
                   shape="box",      # Square shape for Drugs
                   size=40)

    # --- LAYER 2: Proteins (Circles) ---
    # Add Target Nodes
    all_proteins = set(df_dt['target_uniprot_id']).union(set(df_ppi['protein_a_uniprot']) if not df_ppi.empty else set())
    for prot in all_proteins:
        G.add_node(prot, 
                   label=" ",        # Hide labels for cleaner look (hover to see)
                   title=f"Protein: {prot}", 
                   color="#44aaff",  # Blue
                   shape="dot",      # Circle for Proteins
                   size=15)

    # --- EDGE TYPE A: Drug-Target (Solid Lines) ---
    for _, row in df_dt.iterrows():
        G.add_edge(row['generic_name'], row['target_uniprot_id'], 
                   color="#ff4444", # Red edge
                   width=3, 
                   title="Targets")

    # --- EDGE TYPE B: Protein-Protein (Dashed/Gray Lines) ---
    for _, row in df_ppi.iterrows():
        G.add_edge(row['protein_a_uniprot'], row['protein_b_uniprot'], 
                   color="#aaaaaa", # Gray edge
                   width=1, 
                   title="Interacts With")

    # 5. Visualize
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    
    # Configure Physics to separate the clusters
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)
    
    net.from_nx(G)
    output_file = "system_map.html"
    net.show(output_file, notebook=False)
    print(f"✅ System Map saved to: {output_file}")

if __name__ == "__main__":
    # Compare two different families on one map!
    # Painkillers vs Heart Meds
    my_drugs = [
        "ibuprofen", "aspirin", "naproxen",  # NSAIDs
        "atenolol", "sildenafil", "metoprolol" # Cardiovascular
    ]
    draw_system_map(my_drugs)