import numpy as np
from scipy.sparse import csr_matrix, diags

class RWR:
    """
    Random Walk with Restart (RWR) Algorithm.
    Used for network propagation in biological graphs.
    """

    @staticmethod
    def build_transition_matrix(adj_matrix):
        """
        Converts a raw Adjacency Matrix (A) into a column-normalized Transition Matrix (W).
        """
        if not isinstance(adj_matrix, csr_matrix):
            adj_matrix = csr_matrix(adj_matrix)
        
        # Sum along columns (axis=0) to find the degree of each node
        col_sums = np.array(adj_matrix.sum(axis=0)).flatten()
        
        # Avoid division by zero for isolated proteins
        col_sums[col_sums == 0] = 1.0
        
        # Create a diagonal matrix of inverted column sums
        inv_col_sums = diags(1.0 / col_sums)
        
        # Multiply to column-normalize: W = A * D^-1
        transition_matrix = adj_matrix.dot(inv_col_sums)
        return transition_matrix

    @staticmethod
    def calculate_rwr(transition_matrix, initial_vector, restart_prob=0.15, max_iter=100, tol=1e-6):
        """
        Performs Random Walk with Restart on the Transition Matrix.
        Returns the steady-state probability vector (The "Biological Footprint").
        """
        # Ensure initial vector is a probability distribution (sums to 1)
        sum_p0 = np.sum(initial_vector)
        if sum_p0 == 0:
            return initial_vector # Edge case: Drug has no targets
            
        p_0 = initial_vector / sum_p0
        p_t = p_0.copy()
        
        # Iterate until convergence
        for i in range(max_iter):
            # The Core RWR Equation
            p_next = (1 - restart_prob) * transition_matrix.dot(p_t) + (restart_prob * p_0)
            
            # Check convergence (L1 norm difference)
            diff = np.linalg.norm(p_next - p_t, ord=1)
            if diff < tol:
                break
                
            p_t = p_next
            
        return p_t