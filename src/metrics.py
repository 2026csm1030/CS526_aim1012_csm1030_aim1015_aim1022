
# import numpy as np

# from src.graph_utils import approximation_error, is_bipartite_adjacency_matrix


# def evaluate_graph(A, B):
#     """Calculate the main metrics for a graph approximation."""
#     return {
#         "approximation_error": approximation_error(A, B),
#         "is_symmetric": bool(np.allclose(B, B.T)),
#         "is_nonnegative": bool(np.all(B >= 0)),
#         "is_bipartite": is_bipartite_adjacency_matrix(B),
#         "original_edges": int(np.count_nonzero(np.triu(A, k=1))),
#         "retained_edges": int(np.count_nonzero(np.triu(B, k=1))),
#     }


import numpy as np
from src.graph_utils import approximation_error, is_bipartite_adjacency_matrix


def evaluate_graph(A, B):
    """Calculate approximation metrics and diagnose constraint violations."""
    eigenvalues = np.linalg.eigvalsh((B + B.T) / 2)
    sorted_values = np.sort(eigenvalues)

    return {
        "approximation_error": approximation_error(A, B),
        "is_symmetric": bool(np.allclose(B, B.T)),
        "is_nonnegative": bool(np.all(B >= -1e-8)),
        "has_zero_diagonal": bool(np.allclose(np.diag(B), 0, atol=1e-6)),
        "is_bipartite": is_bipartite_adjacency_matrix(B),
        "spectral_symmetry_error": float(
            np.max(np.abs(sorted_values + sorted_values[::-1]))
        ),
        "max_diagonal_value": float(np.max(np.abs(np.diag(B)))),
        "min_edge_weight": float(np.min(B)),
        "original_edges": int(np.count_nonzero(np.triu(A, k=1))),
        "retained_edges": int(np.count_nonzero(np.triu(B, k=1))),
    }
