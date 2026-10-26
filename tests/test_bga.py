
import unittest
import numpy as np

from src.bga_admm import eigenvalue_optimization, admm_bga
from src.graph_generation import generate_wsn_graph
from src.graph_utils import is_bipartite_adjacency_matrix


class TestBGA(unittest.TestCase):

    def test_eigenvalue_optimization(self):
        # Use a symmetric matrix with known eigenvalues.
        M = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0]
        ])

        B = eigenvalue_optimization(M)

        # The projected matrix should be symmetric.
        self.assertTrue(np.allclose(B, B.T))

        # Its eigenvalues should be symmetric around zero.
        eigenvalues = np.linalg.eigvalsh(B)
        eigenvalues = np.sort(eigenvalues)

        self.assertTrue(
            np.allclose(eigenvalues, -eigenvalues[::-1], atol=1e-7)
        )

    def test_admm_output_properties(self):
        A, _ = generate_wsn_graph(
            num_nodes=10,
            radius=0.5,
            seed=42
        )

        B = admm_bga(A, max_iter=500, tolerance=1e-6)

        self.assertTrue(np.allclose(B, B.T))
        self.assertTrue(np.all(B >= -1e-8))

        # A valid loopless adjacency matrix must have a zero diagonal.
        self.assertTrue(np.allclose(np.diag(B), 0, atol=1e-6))

        # This may fail if the ADMM implementation has not converged
        # to a valid bipartite graph.
        self.assertTrue(is_bipartite_adjacency_matrix(B))


if __name__ == "__main__":
    unittest.main()
