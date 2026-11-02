# CS526_aim1012_csm1030_aim1015_aim1022
# Bipartite Graph Approximation by Eigenvalue Optimization

A Python implementation project for experimenting with **ADMM-BGA(1)**, inspired by the paper *Bipartite Graph Approximation by Eigenvalue Optimization*.

This repository contains:
- a weighted wireless sensor network (WSN) graph generator;
- the ADMM-BGA(1) implementation and eigenvalue projection;
- a simple BFS-based Harary-style baseline;
- graph quality and bipartiteness metrics;
- comparison and parameter-ablation experiments;
- Matplotlib scripts for generating plots;
- unit tests.

> **Implementation note:** This is a practical implementation for experimentation. Passing the included tests does not establish an exact reproduction of every detail or guarantee in the paper. Some ablation settings produce outputs that fail the bipartiteness check; those results are retained and reported rather than hidden.

## 1. Project structure

```text
bipartite-graph-approximation/
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── graph_utils.py
│   ├── graph_generation.py
│   ├── bga_admm.py
│   ├── baselines.py
│   └── metrics.py
├── experiments/
│   ├── __init__.py
│   ├── run_comparison.py
│   ├── run_ablation.py
│   └── plot_results.py
├── tests/
│   └── test_bga.py
└── results/
    ├── plots/
    │   ├── comparison_error.png
    │   ├── ablation_max_iter.png
    │   └── ablation_tolerance.png
    └── tables/
        ├── comparison.csv
        └── ablation.csv
```

`results/plots/` and `results/tables/` are generated-output folders. They may be empty before the experiments have been run.

## 2. Requirements

- Python 3.10 or newer recommended
- NumPy
- Matplotlib

Install the dependencies from the project root.

### Install Requirements

```bash
pip install -r requirements.txt
```

## 3. Run the project

Run these commands from the **project root** (`bipartite-graph-approximation/`).

### A. Run the basic example

```bash
python main.py
```

This generates a small WSN graph, runs ADMM-BGA(1) and the Harary-style baseline, and prints their evaluation metrics.

### B. Run unit tests

```bash
python -m unittest tests.test_bga -v
```

The tests check the eigenvalue projection's symmetry properties and selected properties of an ADMM-BGA(1) output.

### C. Run the method comparison experiment

```bash
python -m experiments.run_comparison
```

This runs the methods on the configured graph sizes and writes:

```text
results/tables/comparison.csv
```

The CSV contains the graph size, method, approximation error, bipartiteness result, and retained-edge count.

### D. Run the ablation experiments

```bash
python -m experiments.run_ablation
```

This varies the maximum iteration limit and convergence tolerance. It writes:

```text
results/tables/ablation.csv
```

The table records the varied parameter, its value, the other fixed parameter, approximation error, bipartiteness result, and runtime.

Some configurations may fail the bipartiteness check. This is an observed outcome and should be reported honestly.

### E. Generate plots from the CSV files

After `comparison.csv` and `ablation.csv` exist, run:

```bash
python -m experiments.plot_results
```

This reads the existing CSV files; **it does not rerun the algorithms**. It generates:

```text
results/plots/comparison_error.png
results/plots/ablation_max_iter.png
results/plots/ablation_tolerance.png
```

The figures are saved at 300 DPI for use in a report.

## 4. Recommended run order

For a fresh checkout, run the commands in this order:

```bash
pip install -r requirements.txt
python -m unittest tests.test_bga -v
python main.py
python -m experiments.run_comparison
python -m experiments.run_ablation
python -m experiments.plot_results
```

## 5. Main modules

| File | Purpose |
|---|---|
| `src/graph_generation.py` | Creates synthetic weighted WSN adjacency matrices and node positions. |
| `src/graph_utils.py` | Builds and validates adjacency matrices, computes normalized Frobenius error, and checks bipartiteness. |
| `src/bga_admm.py` | Contains eigenvalue optimization and the ADMM-BGA(1) iteration. |
| `src/baselines.py` | Contains the BFS-based Harary-style approximation baseline. |
| `src/metrics.py` | Collects approximation error, symmetry, nonnegativity, bipartiteness, and edge-count metrics. |
| `main.py` | Runs a small end-to-end example. |
| `experiments/run_comparison.py` | Compares ADMM-BGA(1) and the baseline. |
| `experiments/run_ablation.py` | Tests the effect of iteration limits and convergence tolerance. |
| `experiments/plot_results.py` | Plots the saved CSV experiment results. |
| `tests/test_bga.py` | Contains unit tests for selected algorithm properties. |

## 6. Metrics

The normalized Frobenius approximation error is:

\[
e_{\mathrm{BGA}} = \frac{\|A-B\|_F^2}{\|A\|_F^2},
\]

where `A` is the original adjacency matrix and `B` is the returned approximation. Lower error indicates a closer matrix approximation, but error alone does not establish that `B` is bipartite. The project therefore also checks bipartiteness.

## 7. Configuration notes

- Experiment parameters such as graph size, edge radius, iteration limit, and tolerance are set in the relevant Python files.
- The graph generator places nodes in a unit square. The paper's example uses a particular coordinate scale for its distance-based edge weights; therefore, the paper's `gamma` value may not produce identical weight distributions when used with unit-square coordinates.
- The Harary-style baseline is a simple BFS-based partitioning implementation, not a claim to reproduce every possible Harary-based method.
- The plotting script reads existing CSV files. Rerun the relevant experiment module first if you change its settings and want updated plots.

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'` or `matplotlib`

Activate the virtual environment and run:

```bash
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root and use module execution for package scripts, for example:

```bash
python -m experiments.run_comparison
```

### Plotting fails because a CSV file is missing

Run both experiment scripts first:

```bash
python -m experiments.run_comparison
python -m experiments.run_ablation
python -m experiments.plot_results
```

### An ablation result reports `is_bipartite=False`

This means the returned graph failed the project's bipartiteness check for that run. Do not describe that run as a valid bipartite approximation. Check the iteration limit, tolerance, and ADMM residual/stopping behavior.

## 9. Current validation status

The two included unit tests have passed in the recorded run:

```text
Ran 2 tests
OK
```

This confirms only the properties checked by those tests. It is not a formal proof of convergence or an independent verification that the implementation exactly reproduces the paper.

## 10. Reproducibility

The experiments use fixed random seeds in their current configuration to make the generated graphs repeatable. Runtime measurements can vary across machines and runs. For stronger evaluation, run multiple random seeds and report variability in addition to individual results.
