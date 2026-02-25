import pandas as pd
import networkx as nx
from pyvis.network import Network
from sqlalchemy import create_engine
import os

# Import your newly built Advanced Model
from src.models.tier_1_similarity.advanced_fusion import AdvancedFusionModel

# Database connection
DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"

def generate_fusion_visualization():
    print("🚀 Initializing Fusion Visualization...")
    engine = create_engine(DB_URL)
    model = AdvancedFusionModel()

    # 1. Fetch a small cluster of known interactions to ensure we have red lines
    print("📥 Fetching a sample cluster of interacting drugs...")
    interactions = pd.read_sql("""
        SELECT di.drug_a_id as id1, d1.smiles as smiles1, 
               di.drug_b_id as id2, d2.smiles as smiles2
        FROM drug_interactions di
        JOIN drugs d1 ON di.drug_a_id = d1.id
        JOIN drugs d2 ON di.drug_b_id = d2.id
        WHERE d1.smiles IS NOT NULL AND d2.smiles IS NOT NULL
        LIMIT 15
    """, engine)

    # 2. Extract unique drugs from this cluster
    unique_drugs = {}
    for _, row in interactions.iterrows():
        unique_drugs[row['id1']] = row['smiles1']
        unique_drugs[row['id2']] = row['smiles2']

    drug_ids = list(unique_drugs.keys())
    print(f"🧬 Analyzing {len(drug_ids)} unique drugs...")

    # 3. Build the Graph
    G = nx.Graph()

    # Add Nodes
    for d_id in drug_ids:
        G.add_node(d_id, label=f"Drug {d_id}", title=f"ID: {d_id}", color="#97C2FC")

    # Add RED Edges (Ground Truth Interactions)
    for _, row in interactions.iterrows():
        G.add_edge(row['id1'], row['id2'], color="red", weight=3, title="Known Interaction", dashes=False)

    # Add AI-Predicted Edges (Blue = Chemical, Green = Phenotypic)
    print("🧪 Calculating AI Similarity Scores...")
    for i in range(len(drug_ids)):
        for j in range(i + 1, len(drug_ids)):
            id1, id2 = drug_ids[i], drug_ids[j]
            smiles1, smiles2 = unique_drugs[id1], unique_drugs[id2]
            
            # Use your model to get scores
            s_chem = model.get_chemical_similarity(smiles1, smiles2)
            s_pheno = model.get_phenotypic_similarity(id1, id2)

            # If strong chemical similarity, draw a BLUE dashed line
            if s_chem > 0.15:
                # Only add if edge doesn't already exist to prevent overwriting red lines
                if not G.has_edge(id1, id2):
                    G.add_edge(id1, id2, color="blue", weight=1, title=f"Chem Sim: {s_chem:.2f}", dashes=True)

            # If strong phenotypic (side-effect) similarity, draw a GREEN dotted line
            if s_pheno > 0.02:
                if not G.has_edge(id1, id2):
                    G.add_edge(id1, id2, color="green", weight=2, title=f"Pheno Sim: {s_pheno:.2f}", dashes=[5, 5])

    # 4. Generate Interactive HTML with Pyvis
    print("🎨 Rendering interactive HTML graph...")
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    
    # Add physics for a nice layout
    net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
    
    output_file = "fusion_model_network.html"
    net.write_html(output_file)
    print(f"✅ Success! Open '{output_file}' in your web browser to view the graph.")

if __name__ == "__main__":
    generate_fusion_visualization()