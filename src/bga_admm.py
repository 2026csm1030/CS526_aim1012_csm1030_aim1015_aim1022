
import numpy as np


def eigenvalue_optimization(M):
    """
    Project a symmetric matrix onto the spectral constraint
    that its eigenvalues are symmetric around zero.

    Uses the eigenvalue pairing described in Proposition 1.
    """
    M = (M + M.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Sort eigenvalues from largest to smallest.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n = len(eigenvalues)
    projected = np.zeros(n)

    # Pair the largest and smallest eigenvalues.
    for i in range(n // 2):
        value = (
            eigenvalues[i] - eigenvalues[n - 1 - i]
        ) / 2

        projected[i] = value
        projected[n - 1 - i] = -value

    # For odd-sized matrices, the middle eigenvalue remains zero.
    return eigenvectors @ np.diag(projected) @ eigenvectors.T


def admm_bga(A, max_iter=200, tolerance=1e-5, rho=1.0):
    """
    Preliminary ADMM-BGA(1) implementation.

    A: symmetric, nonnegative adjacency matrix with zero diagonal.
    max_iter: maximum number of iterations.
    tolerance: stopping threshold.
    rho: initial ADMM penalty parameter.
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    if not np.allclose(A, A.T):
        raise ValueError("A must be symmetric.")

    if np.any(A < 0):
        raise ValueError("A must be nonnegative.")

    if not np.allclose(np.diag(A), 0):
        raise ValueError("A must have a zero diagonal.")

    if max_iter < 1 or tolerance <= 0 or rho <= 0:
        raise ValueError("Invalid iteration or convergence parameters.")

    n = A.shape[0]

    B = A.copy()
    Z = A.copy()
    E = np.zeros_like(A)

    Phi = np.zeros_like(A)
    Theta = np.zeros_like(A)

    eta = 1.1
    rho_max = 1e5

    for iteration in range(max_iter):
        B_old = B.copy()
        Z_old = Z.copy()
        E_old = E.copy()

        # Step 1: Update B using the spectral projection.
        M = 0.5 * (
            A - E + Z - (Theta + Phi) / rho
        )
        B = eigenvalue_optimization(M)

        # Step 2: Update Z subject to non-negativity.
        Z = np.maximum(B + Phi / rho, 0)

        # Step 3: Update E subject to 0 <= E <= A.
        E_tilde = A - B - Theta / rho
        E = np.clip(
            (rho / (rho + 2)) * E_tilde,
            0,
            A
        )

        # Step 4: Update dual variables.
        Phi += rho * (B - Z)
        Theta += rho * (B + E - A)

        # Check constraint residuals.
        residual_bz = np.linalg.norm(B - Z, ord="fro")
        residual_bea = np.linalg.norm(B + E - A, ord="fro")

        scale = max(1.0, np.linalg.norm(A, ord="fro"))

        # Measure changes in all three optimization variables.
        change_B = np.linalg.norm(B - B_old, ord="fro")
        change_Z = np.linalg.norm(Z - Z_old, ord="fro")
        change_E = np.linalg.norm(E - E_old, ord="fro")

        residual = max(residual_bz, residual_bea) / scale
        change = max(change_B, change_Z, change_E) / scale

        if residual < tolerance and change < tolerance:
            break

        # Update the penalty parameter as in Algorithm 1.
        rho = min(eta * rho, rho_max)

    # # Return the symmetric, nonnegative auxiliary matrix.
    # B_final = (Z + Z.T) / 2

    # # Remove self-loops from the graph.
    # np.fill_diagonal(B_final, 0.0)

    # return B_final

    # return (Z + Z.T) / 2
    # Symmetrize the nonnegative ADMM output.
    B_final = (Z + Z.T) / 2

    # Remove numerical values where the original graph has no edge.
    B_final[A == 0] = 0.0

    return B_final

#     # Check constraint residuals, not just the change in B.
#     residual_bz = np.linalg.norm(B - Z, ord="fro")
#     residual_bea = np.linalg.norm(
#         B + E - A, ord="fro"
#     )

#     scale = max(1.0, np.linalg.norm(A, ord="fro"))

#     # if max(residual_bz, residual_bea) / scale < tolerance:
#     #     break

#     # # Step 5: Increase the penalty parameter.
#     # rho = min(eta * rho, rho_max)

#     # Measure changes in all three optimization variables.
#     change_B = np.linalg.norm(B - B_old, ord="fro")
#     change_Z = np.linalg.norm(Z - Z_old, ord="fro")
#     change_E = np.linalg.norm(E - E_old, ord="fro")

#     # Stop only when both constraint residuals and variable updates
#     # are sufficiently small relative to the input graph.
#     residual = max(residual_bz, residual_bea) / scale
#     change = max(change_B, change_Z, change_E) / scale

#     if residual < tolerance and change < tolerance:
#         break

#     rho = min(eta * rho, rho_max)

# # # Return the nonnegative auxiliary matrix.
# # # Note: finite iterations do not guarantee exact bipartiteness.
# # return (Z + Z.T) / 2

# print("Final B-Z residual:", np.linalg.norm(B - Z, ord="fro"))
# print("Diagonal before removal:", np.max(np.abs(np.diag(Z))))
# print("Spectral symmetry before diagonal removal:",
#       np.max(np.abs(
#           np.sort(np.linalg.eigvalsh((Z + Z.T) / 2))
#           + np.sort(np.linalg.eigvalsh((Z + Z.T) / 2))[::-1]
#       )))

# # Keep the output symmetric and remove self-loops.
# B_final = (Z + Z.T) / 2
# np.fill_diagonal(B_final, 0.0)

# return B_final


# import numpy as np


# def eigenvalue_optimization(M):
#     """Find the closest matrix with eigenvalues symmetric around zero."""
#     eigenvalues, eigenvectors = np.linalg.eigh(M)

#     # Sort eigenvalues in descending order
#     order = np.argsort(eigenvalues)[::-1]
#     eigenvalues = eigenvalues[order]
#     eigenvectors = eigenvectors[:, order]

#     n = len(eigenvalues)
#     new_values = np.zeros(n)

#     # Make eigenvalues symmetric around zero
#     for i in range(n // 2):
#         value = (eigenvalues[i] - eigenvalues[n - 1 - i]) / 2
#         new_values[i] = value
#         new_values[n - 1 - i] = -value

#     return eigenvectors @ np.diag(new_values) @ eigenvectors.T


# def admm_bga(A, max_iter=200, tolerance=1e-5):
#     """
#     Approximate a graph using ADMM-BGA(1).

#     A: symmetric, nonnegative adjacency matrix.
#     Returns: approximate bipartite adjacency matrix.
#     """
#     n = A.shape[0]

#     B = A.copy()
#     Z = A.copy()
#     E = np.zeros_like(A)

#     Phi = np.zeros_like(A)
#     Theta = np.zeros_like(A)

#     rho = 1.0
#     eta = 1.1
#     rho_max = 1e5

#     for _ in range(max_iter):
#         B_old = B.copy()

#         # Update B using eigenvalue optimization
#         M = 0.5 * (
#             A - E + Z - (Theta + Phi) / rho
#         )
#         B = eigenvalue_optimization(M)

#         # Update Z: enforce nonnegative edge weights
#         Z = np.maximum(B + Phi / rho, 0)

#         # Update E: keep removed edge weights between 0 and A
#         E_tilde = A - B - Theta / rho
#         E = np.clip(
#             (rho / (2 + rho)) * E_tilde,
#             0,
#             A
#         )

#         # Update Lagrange multipliers
#         Phi += rho * (B - Z)
#         Theta += rho * (B + E - A)

#         # Increase penalty parameter
#         rho = min(eta * rho, rho_max)

#         # Stop when the solution changes very little
#         if np.linalg.norm(B - B_old, ord="fro") < tolerance:
#             break

#     # Return a symmetric, nonnegative adjacency matrix
#     B = (Z + Z.T) / 2
#     np.fill_diagonal(B, 0)

#     return B
