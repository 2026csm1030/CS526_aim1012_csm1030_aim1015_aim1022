
import csv
import os

import matplotlib.pyplot as plt


TABLES_DIR = "results/tables"
PLOTS_DIR = "results/plots"


def read_csv(filename):
    """Read a CSV file and return its rows as dictionaries."""
    with open(filename, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_comparison():
    """Compare ADMM-BGA(1) with the Harary-style baseline."""
    rows = read_csv(os.path.join(TABLES_DIR, "comparison.csv"))

    methods = sorted(set(row["method"] for row in rows))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in methods:
        method_rows = [
            row for row in rows if row["method"] == method
        ]
        method_rows.sort(key=lambda row: int(row["nodes"]))

        nodes = [int(row["nodes"]) for row in method_rows]
        errors = [
            float(row["approximation_error"])
            for row in method_rows
        ]

        ax.plot(nodes, errors, marker="o", label=method)

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Normalized Frobenius approximation error")
    ax.set_title("BGA Method Comparison")
    ax.set_xticks([10, 20, 30])
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOTS_DIR, "comparison_error.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ablation(parameter, filename, title, xlabel, log_x=False):
    """Plot approximation error and runtime for one parameter sweep."""
    rows = read_csv(os.path.join(TABLES_DIR, "ablation.csv"))

    rows = [
        row for row in rows
        if row["parameter_varied"] == parameter
    ]
    rows.sort(key=lambda row: float(row["parameter_value"]))

    values = [float(row["parameter_value"]) for row in rows]
    errors = [float(row["approximation_error"]) for row in rows]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    valid = [
        row["is_bipartite"].strip().lower() == "true"
        for row in rows
    ]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Plot approximation error. Mark invalid outputs with crosses.
    ax1.plot(values, errors, marker="o", label="Approximation error")
    for x, y, is_valid in zip(values, errors, valid):
        if not is_valid:
            ax1.scatter(
                [x], [y], marker="x", s=80,
                label="Failed bipartiteness check"
                if "Failed bipartiteness check"
                not in ax1.get_legend_handles_labels()[1]
                else None,
            )

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Normalized approximation error")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    if log_x:
        ax1.set_xscale("log")
        ax1.invert_xaxis()

    # Runtime uses a second y-axis because it has different units.
    ax2 = ax1.twinx()
    ax2.plot(
        values, runtimes, marker="s",
        linestyle="--", label="Runtime",
    )
    ax2.set_ylabel("Runtime (seconds)")

    # Combine legends from both axes.
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOTS_DIR, filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_comparison()

    plot_ablation(
        parameter="max_iter",
        filename="ablation_max_iter.png",
        title="Ablation: Maximum Iterations",
        xlabel="Maximum iterations",
    )

    plot_ablation(
        parameter="tolerance",
        filename="ablation_tolerance.png",
        title="Ablation: Convergence Tolerance",
        xlabel="Convergence tolerance (stricter to the right)",
        log_x=True,
    )

    print("All plots generated in:", PLOTS_DIR)


if __name__ == "__main__":
    main()

