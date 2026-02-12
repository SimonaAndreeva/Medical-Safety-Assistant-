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

def draw_global_map(limit=1000):
    print(f"🌍 Generating Global PPI Map (Limit: {limit} edges)...")
    
    engine = create_engine(settings.DB_URL)
    
    with engine.connect() as conn:
        # Get a slice of the network (LIMIT is crucial here!)
        query = text(f"""
            SELECT protein_a_uniprot, protein_b_uniprot 
            FROM protein_interactions 
            LIMIT {limit}
        """)
        df_ppi = pd.read_sql_query(query, conn)
        print(f"   -> Loaded {len(df_ppi)} interactions.")

    # Build Graph
    G = nx.Graph()
    
    for _, row in df_ppi.iterrows():
        p1 = row['protein_a_uniprot']
        p2 = row['protein_b_uniprot']
        
        # Add nodes with 'Hub' sizing (optional logic could go here)
        G.add_node(p1, color="#4ad0ff", size=5) # Blue nodes
        G.add_node(p2, color="#4ad0ff", size=5)
        
        G.add_edge(p1, p2, color="#333333", width=0.5) # Thin dark edges

    print(f"   -> Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Visualize
    net = Network(height="900px", width="100%", bgcolor="#111111", font_color="white", notebook=False)
    
    # OPTIMIZATION: Use 'BarnesHut' solver for large graphs (stops it from freezing)
    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=95, spring_strength=0.1, damping=0.09)
    
    net.from_nx(G)
    
    output_file = "global_network_map.html"
    net.show(output_file, notebook=False)
    print(f"✅ Map saved to: {output_file}")

if __name__ == "__main__":
    draw_global_map(limit=2000) # Try 2000 edges. If it lags, lower this number.