
from src.metrics import evaluate_graph
from src.baselines import harary_bga
from src.bga_admm import admm_bga
from src.graph_generation import generate_wsn_graph
import csv
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    os.makedirs("results/tables", exist_ok=True)

    output_file = "results/tables/comparison.csv"

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "nodes",
            "method",
            "approximation_error",
            "is_bipartite",
            "retained_edges",
        ])

        for num_nodes in [10, 20, 30]:
            A, _ = generate_wsn_graph(
                num_nodes=num_nodes,
                radius=0.4,
                seed=42,
            )

            # methods = {
            #     "ADMM-BGA(1)": admm_bga(A),
            #     "Harary-style": harary_bga(A),
            # }

            methods = {
                # "ADMM-BGA(1)": admm_bga(
                #     A, max_iter=1000, tolerance=1e-5
                # ),
                "ADMM-BGA(1)": admm_bga(
                    A,
                    max_iter=5000,
                    tolerance=1e-8,
                ),
                "Harary-style": harary_bga(A)
            }

            for name, B in methods.items():
                metrics = evaluate_graph(A, B)

                # print("Maximum B - A:", np.max(B - A))
                # print("Edges outside original graph:", np.count_nonzero(
                #     np.triu((B > 1e-8) & (A <= 1e-8), k=1)
                # ))

                # if name == "ADMM-BGA(1)" and num_nodes == 30:
                #     outside = np.triu((B > 1e-8) & (A <= 1e-8), k=1)
                #     weights = B[outside]

                #     print("Number of outside edges:", len(weights))
                #     print("Largest outside-edge weight:",
                #           weights.max() if len(weights) else 0)
                #     print("Smallest outside-edge weight:",
                #           weights.min() if len(weights) else 0)
                #     print("First 10 outside-edge weights:",
                #           np.sort(weights)[:10])

                # print(metrics)

                writer.writerow([
                    num_nodes,
                    name,
                    metrics["approximation_error"],
                    metrics["is_bipartite"],
                    metrics["retained_edges"],
                ])

                print(
                    f"Nodes: {num_nodes}, "
                    f"Method: {name}, "
                    f"Error: {metrics['approximation_error']:.4f}, "
                    f"Bipartite: {metrics['is_bipartite']}"
                )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

