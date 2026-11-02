
import numpy as np


def generate_wsn_graph(num_nodes=30, radius=0.4, gamma=10, seed=42):
    """
    Generate a weighted wireless sensor network graph.

    Nodes are placed randomly in a unit square.
    Edges exist when the distance between nodes is <= radius.
    Edge weights follow the paper's distance-based formula.
    """
    rng = np.random.default_rng(seed)

    # Random node positions in [0, 1] x [0, 1]
    positions = rng.random((num_nodes, 2))

    # Calculate pairwise Euclidean distances
    differences = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)

    # Create weighted adjacency matrix
    A = np.exp(-(distances ** 2) / (2 * gamma))
    A[distances > radius] = 0.0

    # Remove self-loops
    np.fill_diagonal(A, 0.0)

    return A, positions
