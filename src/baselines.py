
import numpy as np
from collections import deque


def harary_bga(A):
    """
    Simple Harary-style bipartite approximation.

    Retains edges between opposite partitions and removes
    edges within the same partition.
    """
    n = A.shape[0]
    colors = [-1] * n

    for start in range(n):
        if colors[start] != -1:
            continue

        colors[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()

            for v in range(n):
                if A[u, v] > 0 and colors[v] == -1:
                    colors[v] = 1 - colors[u]
                    queue.append(v)

    B = A.copy()

    for i in range(n):
        for j in range(n):
            if colors[i] == colors[j]:
                B[i, j] = 0.0

    np.fill_diagonal(B, 0.0)

    return B
