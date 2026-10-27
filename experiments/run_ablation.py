
import csv
import os
import time

from src.graph_generation import generate_wsn_graph
from src.bga_admm import admm_bga
from src.metrics import evaluate_graph


def main():
    os.makedirs("results/tables", exist_ok=True)

    output_file = "results/tables/ablation.csv"

    # Use the same graph for every experiment.
    A, _ = generate_wsn_graph(
        num_nodes=20,
        radius=0.4,
        seed=42,
    )

    results = []

    # Experiment 1: vary the maximum iteration count.
    fixed_tolerance = 1e-8

    for max_iter in [200, 500, 1000, 2000, 5000]:
        start = time.perf_counter()

        B = admm_bga(
            A,
            max_iter=max_iter,
            tolerance=fixed_tolerance,
        )

        runtime = time.perf_counter() - start
        metrics = evaluate_graph(A, B)

        row = [
            "max_iter",
            max_iter,
            fixed_tolerance,
            metrics["approximation_error"],
            metrics["is_bipartite"],
            runtime,
        ]
        results.append(row)

        print(
            f"max_iter={max_iter}, "
            f"error={metrics['approximation_error']:.6f}, "
            f"bipartite={metrics['is_bipartite']}, "
            f"runtime={runtime:.3f}s"
        )

    # Experiment 2: vary tolerance.
    fixed_max_iter = 5000

    for tolerance in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
        start = time.perf_counter()

        B = admm_bga(
            A,
            max_iter=fixed_max_iter,
            tolerance=tolerance,
        )

        runtime = time.perf_counter() - start
        metrics = evaluate_graph(A, B)

        row = [
            "tolerance",
            tolerance,
            fixed_max_iter,
            metrics["approximation_error"],
            metrics["is_bipartite"],
            runtime,
        ]
        results.append(row)

        print(
            f"tolerance={tolerance:.0e}, "
            f"error={metrics['approximation_error']:.6f}, "
            f"bipartite={metrics['is_bipartite']}, "
            f"runtime={runtime:.3f}s"
        )

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "parameter_varied",
            "parameter_value",
            "other_parameter_fixed",
            "approximation_error",
            "is_bipartite",
            "runtime_seconds",
        ])
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
