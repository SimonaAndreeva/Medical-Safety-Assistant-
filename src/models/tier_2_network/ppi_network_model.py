import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import networkx as nx

# --- YOUR EXISTING MATH ENGINE (Kept Intact) ---
class RWR:
    """Random Walk with Restart (RWR) Algorithm Core Math."""
    @staticmethod
    def build_transition_matrix(adj_matrix):
        if not isinstance(adj_matrix, csr_matrix):
            adj_matrix = csr_matrix(adj_matrix)
        col_sums = np.array(adj_matrix.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1.0
        inv_col_sums = diags(1.0 / col_sums)
        return adj_matrix.dot(inv_col_sums)

    @staticmethod
    def calculate_rwr(transition_matrix, initial_vector, restart_prob=0.15, max_iter=100, tol=1e-6):
        sum_p0 = np.sum(initial_vector)
        if sum_p0 == 0:
            return initial_vector 
            
        p_0 = initial_vector / sum_p0
        p_t = p_0.copy()
        
        for i in range(max_iter):
            p_next = (1 - restart_prob) * transition_matrix.dot(p_t) + (restart_prob * p_0)
            diff = np.linalg.norm(p_next - p_t, ord=1)
            if diff < tol:
                break
            p_t = p_next
        return p_t

# --- THE NEW BIOLOGICAL WRAPPER (Tier II) ---
class PPINetworkModel:
    def __init__(self, db_url="postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"):
        self.engine = create_engine(db_url)
        self.rwr_cache = {} # Cache to save computation time
        
        print("Building Protein-Protein Interaction (PPI) Network...")
        self._build_network()

    def _build_network(self):
        """Fetches PPIs and prepares the transition matrix."""
        # 1. Fetch protein edges using the correct schema columns
        ppi_data = pd.read_sql("SELECT protein_a_uniprot, protein_b_uniprot FROM protein_interactions", self.engine)
        
        # 2. Build graph to get nodes and adjacency matrix
        G = nx.Graph()
        # Update the zip function to use the correct columns
        edges = list(zip(ppi_data['protein_a_uniprot'], ppi_data['protein_b_uniprot']))
        G.add_edges_from(edges)
        
        # 3. Create index mappings for the matrix
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        
        # 4. Build Transition Matrix using your RWR class
        adj_matrix = nx.to_scipy_sparse_array(G, nodelist=self.nodes)
        self.transition_matrix = RWR.build_transition_matrix(adj_matrix)
        
        # 5. Pre-load drug targets to create seed vectors
        self.drug_targets = pd.read_sql("SELECT drug_id, target_uniprot_id FROM drug_targets", self.engine)
        
        print(f"PPI Network built with {self.num_nodes} proteins and {len(edges)} interactions.")

    def get_drug_seed_vector(self, drug_id):
        """Creates the initial probability vector (p0) for a drug."""
        targets = self.drug_targets[self.drug_targets['drug_id'] == drug_id]['target_uniprot_id'].tolist()
        
        p0 = np.zeros(self.num_nodes)
        valid_targets = 0
        
        for target in targets:
            if target in self.node_to_idx:
                idx = self.node_to_idx[target]
                p0[idx] = 1.0
                valid_targets += 1
                
        return p0, valid_targets

    def get_network_similarity(self, drug1_id, drug2_id):
        """Calculates Cosine Similarity between the RWR footprints of two drugs."""
        def get_rwr_footprint(d_id):
            if d_id in self.rwr_cache:
                return self.rwr_cache[d_id]
                
            p0, target_count = self.get_drug_seed_vector(d_id)
            if target_count == 0:
                # If drug has no known targets in our PPI, return zeros
                footprint = np.zeros(self.num_nodes)
            else:
                # Run your math engine
                footprint = RWR.calculate_rwr(self.transition_matrix, p0)
                
            self.rwr_cache[d_id] = footprint
            return footprint

        # Get footprints
        fp1 = get_rwr_footprint(drug1_id)
        fp2 = get_rwr_footprint(drug2_id)
        
        # If either drug has no targets, network similarity is 0
        if np.sum(fp1) == 0 or np.sum(fp2) == 0:
            return 0.0
            
        # Cosine similarity between the steady-state probabilities
        sim = cosine_similarity(fp1.reshape(1, -1), fp2.reshape(1, -1))[0][0]
        return sim

if __name__ == "__main__":
    # Quick test to ensure it runs
    model = PPINetworkModel()
    
    # Replace these IDs with actual drug IDs from your database that have targets
    test_d1 = 1  
    test_d2 = 2  
    
    sim_score = model.get_network_similarity(test_d1, test_d2)
    print(f"\nNetwork Topology Similarity Score: {sim_score:.4f}")