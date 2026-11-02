
import numpy as np


def create_adjacency_matrix(num_nodes, edges):
    """
    Create a symmetric adjacency matrix from weighted edges.

    edges: list of (node1, node2, weight)
    """
    A = np.zeros((num_nodes, num_nodes))

    for u, v, weight in edges:
        A[u, v] = weight
        A[v, u] = weight

    return A


def is_valid_adjacency_matrix(A):
    """Check whether A is square, symmetric, nonnegative, and loopless."""
    return (
        A.ndim == 2
        and A.shape[0] == A.shape[1]
        and np.allclose(A, A.T)
        and np.all(A >= 0)
        and np.allclose(np.diag(A), 0)
    )


def approximation_error(A, B):
    """Compute the normalized Frobenius approximation error."""
    denominator = np.linalg.norm(A, ord="fro")

    if denominator == 0:
        return 0.0

    return np.linalg.norm(A - B, ord="fro") ** 2 / denominator**2


def is_bipartite_adjacency_matrix(B, tolerance=1e-8):
    """Check bipartiteness using a two-coloring traversal."""
    n = B.shape[0]
    colors = [-1] * n

    for start in range(n):
        if colors[start] != -1:
            continue

        colors[start] = 0
        queue = [start]

        while queue:
            u = queue.pop(0)

            for v in range(n):
                if B[u, v] > tolerance:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        return False

    return True
