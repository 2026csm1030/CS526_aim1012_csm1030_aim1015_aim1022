
import numpy as np

from src.graph_utils import (
    create_adjacency_matrix,
    is_valid_adjacency_matrix,
    approximation_error,
    is_bipartite_adjacency_matrix,
)

edges = [
    (0, 1, 2.0),
    (1, 2, 1.0),
]

A = create_adjacency_matrix(3, edges)

print("Adjacency matrix:")
print(A)

print("Valid adjacency matrix:", is_valid_adjacency_matrix(A))
print("Self approximation error:", approximation_error(A, A))
print("Bipartite graph:", is_bipartite_adjacency_matrix(A))
