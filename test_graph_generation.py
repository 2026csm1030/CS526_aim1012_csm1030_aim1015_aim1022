
from src.graph_generation import generate_wsn_graph
from src.graph_utils import is_valid_adjacency_matrix

A, positions = generate_wsn_graph(num_nodes=30, radius=0.4, seed=42)

print("Adjacency matrix shape:", A.shape)
print("Node positions shape:", positions.shape)
print("Valid adjacency matrix:", is_valid_adjacency_matrix(A))
print("Number of edges:", int((A > 0).sum() / 2))
