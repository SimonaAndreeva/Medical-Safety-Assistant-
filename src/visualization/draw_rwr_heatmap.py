import sys
import os

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pyvis.network import Network
from sqlalchemy import create_engine, text
from src.config import settings

def draw_rwr_heatmap(drug_name, top_n=100):
    print(f"🔥 Generating RWR Heatmap for: '{drug_name}'...")

    # 1. Connect to DB and find Drug ID
    engine = create_engine(settings.DB_URL)
    with engine.connect() as conn:
        query_drug = text("SELECT id, generic_name FROM drugs WHERE generic_name ILIKE :name LIMIT 1")
        df_drug = pd.read_sql_query(query_drug, conn, params={"name": f"%{drug_name}%"})
        
        if df_drug.empty:
            print(f"❌ Drug '{drug_name}' not found in database.")
            return
            
        drug_id = int(df_drug.iloc[0]['id'])
        real_name = df_drug.iloc[0]['generic_name']
        print(f"   -> Found Drug: {real_name} (ID: {drug_id})")

    # 2. Load the RWR "Brain" (The continuous embeddings)
    try:
        with open(settings.NETWORK_FEATURES, 'rb') as f:
            df_rwr = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: RWR features not found. Run 'build_network.py' first.")
        return

    if drug_id not in df_rwr.index:
        print(f"❌ No RWR data calculated for {real_name}.")
        return

    # 3. Get the Top N Proteins by RWR Score
    print(f"   -> Extracting the top {top_n} most affected proteins...")
    drug_rwr_profile = df_rwr.loc[drug_id]
    
    # Sort descending and get the top N
    top_proteins = drug_rwr_profile.sort_values(ascending=False).head(top_n)
    
    # Normalize scores between 0 and 1 strictly for visual sizing/coloring
    max_score = top_proteins.max()
    min_score = top_proteins.min()
    
    # 4. Fetch interactions between these specific proteins
    protein_list = tuple(top_proteins.index.tolist())
    with engine.connect() as conn:
        query_ppi = text(f"""
            SELECT protein_a_uniprot, protein_b_uniprot 
            FROM protein_interactions 
            WHERE protein_a_uniprot IN {protein_list} 
              AND protein_b_uniprot IN {protein_list}
        """)
        df_ppi = pd.read_sql_query(query_ppi, conn)
    
    print(f"   -> Found {len(df_ppi)} connections within this biological neighborhood.")

    # 5. Build the Heatmap Graph
    G = nx.Graph()

    # Add the Drug Node in the center
    G.add_node(real_name, size=50, color="#ffffff", title="The Drug", shape="star")

    # Add Protein Nodes with Heatmap Logic
    for prot, score in top_proteins.items():
        if score == 0: continue
        
        # Calculate visual intensity (0.0 to 1.0)
        intensity = (score - min_score) / (max_score - min_score + 1e-9)
        
        # Determine Size (10 to 40)
        node_size = 10 + (30 * intensity)
        
        # Determine Color (Red -> Orange -> Yellow -> Blue)
        if intensity > 0.8:
            color = "#ff0000" # Red (Direct targets usually)
        elif intensity > 0.4:
            color = "#ffaa00" # Orange
        elif intensity > 0.1:
            color = "#ffff00" # Yellow
        else:
            color = "#00aaff" # Light Blue (The distant ripple effect)

        hover_text = f"Protein: {prot}<br>RWR Probability: {score:.6f}"
        
        G.add_node(prot, size=node_size, color=color, title=hover_text)
        
        # Draw a faint edge from the drug to the highly affected proteins
        if intensity > 0.8:
            G.add_edge(real_name, prot, color="#ffffff", value=2)

    # Add PPI Edges
    for _, row in df_ppi.iterrows():
        p1, p2 = row['protein_a_uniprot'], row['protein_b_uniprot']
        if p1 in G and p2 in G:
            G.add_edge(p1, p2, color="#444444", value=0.5) # Dark grey lines

    # 6. Render with PyVis
    net = Network(height="800px", width="100%", bgcolor="#111111", font_color="white", notebook=False)
    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150)
    net.from_nx(G)
    
    output_file = f"rwr_heatmap_{real_name}.html"
    net.show(output_file, notebook=False)
    print(f"✅ Heatmap saved to: {output_file}")

if __name__ == "__main__":
    draw_rwr_heatmap("sildenafil", top_n=100)