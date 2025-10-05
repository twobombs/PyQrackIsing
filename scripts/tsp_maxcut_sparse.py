# MAXCUT from TSP (Sparse)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import tsp_maxcut_sparse
import numpy as np
import os
import sys
import time
from scipy.sparse import lil_matrix


def generate_adjacency(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)

    lil = lil_matrix((n_nodes, n_nodes), dtype=np.float32)

    for u in range(n_nodes >> 1):
        for v in range(u + 1, n_nodes >> 1):
            lil[u, v] = np.random.random()

    for u in range(n_nodes >> 1, n_nodes):
        for v in range(u + 1, n_nodes):
            lil[u, v] = np.random.random()

    return lil.tocsr()


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    multi_start = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    k_neighbors = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")

    start = time.perf_counter()
    G_m = generate_adjacency(n_nodes=n_nodes, seed=seed)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the adjacency matrix (statement of the problem itself)")

    start = time.perf_counter()
    bit_string, cut_value, partition, energy = tsp_maxcut_sparse(G_m)
    seconds = time.perf_counter() - start

    print(f"Seconds to MAXCUT solution: {seconds}")
    print(f"Partition: {bit_string}")
    print(f"Cut value: {cut_value}")
    print(
        "(The average randomized and normalized separation between each and every node is about 0.5.)"
    )
