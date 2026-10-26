
import numpy as np

from src.bga_admm import admm_bga
from src.graph_generation import generate_wsn_graph
from src.graph_utils import is_bipartite_adjacency_matrix
from src.baselines import harary_bga


def find_conflicting_edges(B, tolerance=1e-8):
    """Return edges that violate a BFS two-coloring."""
    n = B.shape[0]
    colors = [-1] * n
    parent = [-1] * n
    queue = []

    for start in range(n):
        if colors[start] != -1:
            continue

        colors[start] = 0
        queue = [start]

        while queue:
            u = queue.pop(0)

            for v in range(n):
                if B[u, v] <= tolerance:
                    continue

                if colors[v] == -1:
                    colors[v] = 1 - colors[u]
                    parent[v] = u
                    queue.append(v)

                elif colors[v] == colors[u]:
                    return (u, v, B[u, v], colors)

    return None


def test_thresholds(A, B):
    """Measure how thresholding small weights affects graph validity."""
    print("Threshold | Bipartite | Approximation error | Removed edges")

    denominator = np.linalg.norm(A, ord="fro") ** 2

    for threshold in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
        candidate = B.copy()
        candidate[candidate < threshold] = 0.0

        error = (
            np.linalg.norm(A - candidate, ord="fro") ** 2
            / denominator
            if denominator > 0 else 0.0
        )

        removed = int(
            np.count_nonzero(np.triu(B, k=1))
            - np.count_nonzero(np.triu(candidate, k=1))
        )

        print(
            f"{threshold:.0e} | "
            f"{is_bipartite_adjacency_matrix(candidate)} | "
            f"{error:.6f} | {removed}"
        )


def main():
    for n in [10, 20, 30]:
        A, _ = generate_wsn_graph(
            num_nodes=n, radius=0.4, seed=42
        )
        B = admm_bga(A, max_iter=1000, tolerance=1e-5)

        B_valid = harary_bga(B)

        print("ADMM bipartite:", is_bipartite_adjacency_matrix(B))
        print("Corrected bipartite:", is_bipartite_adjacency_matrix(B_valid))
        print(
            "ADMM error:",
            np.linalg.norm(A - B, ord="fro") ** 2
            / np.linalg.norm(A, ord="fro") ** 2
        )
        print(
            "Corrected error:",
            np.linalg.norm(A - B_valid, ord="fro") ** 2
            / np.linalg.norm(A, ord="fro") ** 2
        )

        test_thresholds(A, B)

        conflict = find_conflicting_edges(B)

        print(f"\nNodes: {n}")
        print("Bipartite:", is_bipartite_adjacency_matrix(B))

        if conflict is None:
            print("No conflicting edge found.")
        else:
            u, v, weight, colors = conflict
            print(f"Conflicting edge: ({u}, {v}), weight={weight:.10f}")
            print("Endpoint colors:", colors[u], colors[v])


if __name__ == "__main__":
    main()
