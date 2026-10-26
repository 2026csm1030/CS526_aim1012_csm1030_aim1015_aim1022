
from src.graph_generation import generate_wsn_graph
from src.baselines import harary_bga
from src.graph_utils import (
    approximation_error,
    is_bipartite_adjacency_matrix,
)

A, _ = generate_wsn_graph(num_nodes=10, radius=0.5)

B = harary_bga(A)

print("Approximation error:", approximation_error(A, B))
print("Bipartite:", is_bipartite_adjacency_matrix(B))
