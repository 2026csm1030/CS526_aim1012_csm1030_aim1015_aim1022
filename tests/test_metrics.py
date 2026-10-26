
from src.graph_generation import generate_wsn_graph
from src.baselines import harary_bga
from src.metrics import evaluate_graph

A, _ = generate_wsn_graph(num_nodes=10, radius=0.5)
B = harary_bga(A)

print(evaluate_graph(A, B))
