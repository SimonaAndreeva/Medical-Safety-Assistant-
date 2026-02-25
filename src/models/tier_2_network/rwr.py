import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.metrics.pairwise import cosine_similarity


class RWR:
    """
    Tier 2 — Random Walk with Restart (RWR) Engine.

    Responsibility: Network graph propagation and biological embedding similarity.
    Operates on the Protein-Protein Interaction (PPI) graph.

    Methods:
        build_transition_matrix — column-normalizes a sparse adjacency matrix (W = A · D⁻¹)
        calculate_rwr           — iterative RWR to produce steady-state biological footprints
        calculate_cosine        — cosine similarity between two RWR embedding vectors
    """

    @staticmethod
    def build_transition_matrix(adj_matrix):
        """
        Converts a raw adjacency matrix A into a column-normalized
        transition matrix W = A · D⁻¹.

        W[i][j] = probability a random walker at node j moves to node i.
        Each column sums to 1 (column-stochastic).
        """
        if not isinstance(adj_matrix, csr_matrix):
            adj_matrix = csr_matrix(adj_matrix)

        # Column sums = degree of each node
        col_sums = np.array(adj_matrix.sum(axis=0)).flatten()

        # Prevent division by zero for isolated nodes (proteins with no PPI edges)
        col_sums[col_sums == 0] = 1.0

        # D⁻¹ as a sparse diagonal matrix (avoids dense memory allocation)
        inv_col_sums = diags(1.0 / col_sums)

        # W = A · D⁻¹
        return adj_matrix.dot(inv_col_sums)

    @staticmethod
    def calculate_rwr(transition_matrix, initial_vector, restart_prob=0.15, max_iter=100, tol=1e-6):
        """
        Performs Random Walk with Restart on the transition matrix.
        Returns the steady-state probability vector — the 'biological footprint'.

        Core equation (iterated until convergence):
            p(t+1) = (1 - r) * W * p(t) + r * p0

        Args:
            transition_matrix: Column-normalized W from build_transition_matrix
            initial_vector:    Seed vector p0 — probability mass on drug's target proteins
            restart_prob:      r — probability of teleporting back to seed (default 0.15)
            max_iter:          Maximum iterations before forced stop
            tol:               L1-norm convergence threshold (default 1e-6)
        """
        sum_p0 = np.sum(initial_vector)

        # Edge case: drug has no known targets in the PPI graph
        if sum_p0 == 0:
            return initial_vector

        # Normalize seed to a valid probability distribution (sums to 1)
        p_0 = initial_vector / sum_p0
        p_t = p_0.copy()

        for _ in range(max_iter):
            p_next = (1 - restart_prob) * transition_matrix.dot(p_t) + (restart_prob * p_0)

            # L1-norm: total probability mass shifted this iteration
            diff = np.linalg.norm(p_next - p_t, ord=1)
            if diff < tol:
                break

            p_t = p_next

        return p_t

    @staticmethod
    def calculate_cosine(target_vector, all_vectors):
        """
        Computes Cosine Similarity between RWR steady-state probability vectors.

        Lives in Tier 2 because it operates on the continuous embedding output
        of calculate_rwr — not on raw chemical fingerprints.

        Formula: cos(p, q) = (p · q) / (||p||₂ * ||q||₂)
        """
        return cosine_similarity(target_vector, all_vectors).flatten()