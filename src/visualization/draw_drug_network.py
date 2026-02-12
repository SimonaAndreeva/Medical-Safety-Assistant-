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

def draw_mechanism(drug_name):
    print(f"🎨 Generating Biological Graph for: {drug_name}...")
    
    engine = create_engine(settings.DB_URL)
    
    with engine.connect() as conn:
        # 1. Find Drug ID
        query_drug = text("SELECT id, generic_name FROM drugs WHERE generic_name ILIKE :name")
        df_drug = pd.read_sql_query(query_drug, conn, params={"name": f"%{drug_name}%"})
        
        if df_drug.empty:
            print("❌ Drug not found.")
            return
            
        # FIX: Convert numpy.int64 to standard python int
        drug_id = int(df_drug.iloc[0]['id']) 
        real_name = df_drug.iloc[0]['generic_name']
        print(f"   -> Found Drug: {real_name} (ID: {drug_id})")

        # 2. Get Direct Targets (Drug -> Protein)
        query_targets = text("SELECT target_uniprot_id FROM drug_targets WHERE drug_id = :did")
        df_targets = pd.read_sql_query(query_targets, conn, params={"did": drug_id})
        target_list = df_targets['target_uniprot_id'].tolist()
        
        print(f"   -> Found {len(target_list)} direct targets.")

        # 3. Get Protein-Protein Interactions (Protein <-> Neighbor)
        # We only get interactions involving our targets
        if not target_list:
            print("⚠️ No targets found. Graph will be empty.")
            return

        formatted_targets = tuple(target_list)
        # Handle tuple syntax for single item (Python quirk: (1) is int, (1,) is tuple)
        if len(target_list) == 1:
            formatted_targets = f"('{target_list[0]}')"
        
        # Use text() for safe IN clause usage (though params is better, IN is tricky with lists)
        query_ppi = text(f"""
            SELECT protein_a_uniprot, protein_b_uniprot 
            FROM protein_interactions 
            WHERE protein_a_uniprot IN {formatted_targets} 
               OR protein_b_uniprot IN {formatted_targets}
            LIMIT 500
        """)
        
        df_ppi = pd.read_sql_query(query_ppi, conn)
        print(f"   -> Found {len(df_ppi)} interactions (Neighbors).")

    # 4. Build the Graph
    G = nx.Graph()
    
    # Add Drug Node (The Center)
    G.add_node(real_name, color="#ff0000", title="Drug", size=30, group="drug") # Red
    
    # Add Target Nodes (Level 1)
    for t in target_list:
        G.add_node(t, color="#00ff00", title="Direct Target", size=20, group="target") # Green
        G.add_edge(real_name, t, color="#000000") # Black edge

    # Add Neighbor Nodes (Level 2 - The PPI)
    for _, row in df_ppi.iterrows():
        p1, p2 = row['protein_a_uniprot'], row['protein_b_uniprot']
        
        # Add nodes if they don't exist yet (Gray for neighbors)
        if p1 not in G: G.add_node(p1, color="#aaaaaa", size=10, group="neighbor")
        if p2 not in G: G.add_node(p2, color="#aaaaaa", size=10, group="neighbor")
        
        # Add edge (Gray)
        G.add_edge(p1, p2, color="#dddddd")

    # 5. Visualize with PyVis
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.from_nx(G)
    
    # Add Physics (so it bounces nicely)
    net.toggle_physics(True)
    
    output_file = f"network_{drug_name}.html"
    net.show(output_file, notebook=False)
    print(f"✅ Graph saved to: {output_file}")
    print("   (Open this file in your browser to see the interactive model!)")

if __name__ == "__main__":
    # Test with a drug that has targets
    draw_mechanism("sildenafil")