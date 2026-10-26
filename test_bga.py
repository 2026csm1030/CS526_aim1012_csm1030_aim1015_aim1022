
from src.graph_generation import generate_wsn_graph
from src.bga_admm import admm_bga
from src.graph_utils import (
    approximation_error,
    is_bipartite_adjacency_matrix,
)

A, _ = generate_wsn_graph(num_nodes=10, radius=0.5)

B = admm_bga(A)

print("Original graph shape:", A.shape)
print("Approximation error:", approximation_error(A, B))
print("Symmetric:", __import__("numpy").allclose(B, B.T))
print("Nonnegative:", (B >= 0).all())
print("Bipartite:", is_bipartite_adjacency_matrix(B))
