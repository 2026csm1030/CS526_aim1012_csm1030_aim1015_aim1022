

# import numpy as np
# from eigenvalue_optimization import eigenvalue_projection

# # A triangle graph: every vertex connects to the other two
# A = np.array([
#     [0, 1, 1],
from src.graph_generation import generate_wsn_graph
from src.bga_admm import admm_bga
from src.baselines import harary_bga
from src.metrics import evaluate_graph


def main():
    # Generate a sample graph
    A, positions = generate_wsn_graph(
        num_nodes=10,
        radius=0.5,
        seed=42,
    )

    # Run both methods
    B_admm = admm_bga(A)
    B_harary = harary_bga(A)

    # Evaluate results
    print("Original graph edges:", int((A > 0).sum() / 2))

    print("\n--- ADMM-BGA(1) ---")
    for metric, value in evaluate_graph(A, B_admm).items():
        print(f"{metric}: {value}")

    print("\n--- Harary-style baseline ---")
    for metric, value in evaluate_graph(A, B_harary).items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()

#     [1, 0, 1],
#     [1, 1, 0]
# ], dtype=float)

# B, original_eigenvalues, new_eigenvalues = (
#     eigenvalue_projection(A)
# )

# print("Original adjacency matrix:")
# print(A)

# print("\nOriginal eigenvalues (descending):")
# print(original_eigenvalues)

# print("\nAdjusted eigenvalues:")
# print(new_eigenvalues)

# print("\nReconstructed matrix:")
# print(np.round(B, 4))

# print("\nIs B symmetric?")
# print(np.allclose(B, B.T))

# print("\nAre eigenvalues symmetric around zero?")
# print(np.allclose(
#     new_eigenvalues,
#     -new_eigenvalues[::-1]
# ))
