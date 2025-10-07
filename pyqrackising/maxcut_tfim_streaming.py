import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import fix_cdf, get_cut, init_thresholds, maxcut_hamming_cdf, opencl_context, probability_by_hamming_weight

IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


@njit
def update_repulsion_choice(G_func, nodes, max_weight, weights, n, used, node):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for nbr in range(n):
        if used[nbr]:
            continue
        weights[nbr] *= max(1.1920928955078125e-7, 1 - G_func(nodes[node], nodes[nbr]) / max_weight)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_func, nodes, max_weight, weights, n, m, shot):
    """
    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency_data, adjacency_rows: CSR-format sparse adjacency data
    weights: float64 array of shape (n,)
    """

    weights = weights.copy()
    used = np.zeros(n, dtype=np.bool_) # False = available, True = used

    # Update answer and weights
    update_repulsion_choice(G_func, nodes, max_weight, weights, n, used, shot % n)

    for _ in range(1, m):
        # Count available
        total_w = 0.0
        for i in range(n):
            if used[i]:
                continue
            total_w += weights[i]

        # Normalize & sample
        r = np.random.rand()
        cum = 0.0
        node = -1
        for i in range(n):
            if used[i]:
                continue
            cum += weights[i]
            if (total_w * r) < cum:
                node = i
                break

        if node == -1:
            node = 0
            while used[node]:
                node += 1

        # Select node
        used[node] = True

        # Update answer and weights
        update_repulsion_choice(G_func, nodes, max_weight, weights, n, used, node)

    return used


@njit
def compute_energy(sample, G_func, nodes, n_qubits):
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            val = G_func(nodes[u], nodes[v])
            energy += val if sample[u] == sample[v] else -val

    return energy


@njit(parallel=True)
def sample_for_solution(G_func, nodes, max_weight, shots, thresholds, degrees_sum, weights, n, dtype):
    shots = max(1, shots >> 1)

    solutions = np.empty((shots, n), dtype=np.bool_)
    energies = np.empty(shots, dtype=dtype)

    best_solution = solutions[0]
    best_energy = float("inf")

    improved = True
    while improved:
        improved = False
        for s in prange(shots):
            # First dimension: Hamming weight
            mag_prob = np.random.random()
            m = 0
            while thresholds[m] < mag_prob:
                m += 1
            m += 1

            # Second dimension: permutation within Hamming weight
            sample = local_repulsion_choice(G_func, nodes, max_weight, weights, n, m, s)
            solutions[s] = sample
            energies[s] = compute_energy(sample, G_func, nodes, n)

        best_index = np.argmin(energies)
        energy = energies[best_index]
        if energy < best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    best_value = 0.0
    for u in range(n):
        for v in range(u + 1, n):
            if best_solution[u] != best_solution[v]:
                best_value += G_func(nodes[u], nodes[v])

    return best_solution, best_value


@njit(parallel=True)
def init_J_and_z(G_func, nodes, dtype):
    n_qubits = len(nodes)
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=dtype)
    J_max = -float("inf")
    G_max = -float("inf")
    for n in prange(n_qubits):
        degree = 0
        J = 0.0
        for m in range(n_qubits):
            val = G_func(nodes[n], nodes[m])
            if val != 0.0:
                degree += 1
                J += val
            G_max = max(val, G_max)
        J = -J / degree if degree > 0 else 0
        degrees[n] = degree
        J_eff[n] = J
        J_abs = abs(J)
        J_max = max(J_abs, J_max)
    J_eff /= J_max

    return J_eff, degrees, G_max


@njit
def cpu_footer(shots, quality, n_qubits, G_func, nodes, dtype, epsilon):
    J_eff, degrees, G_max = init_J_and_z(G_func, nodes, dtype)
    hamming_prob = init_thresholds(n_qubits, dtype)

    maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, hamming_prob, dtype)

    max_weight = degrees.sum()
    degrees = None
    J_eff = 1.0 / (1.0 + epsilon - J_eff)

    best_solution, best_value = sample_for_solution(G_func, nodes, G_max, shots, hamming_prob, max_weight, J_eff, n_qubits, dtype)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


@njit
def gpu_footer(shots, n_qubits, G_func, nodes, G_max, weights, degrees, hamming_prob, max_weight, dtype):
    fix_cdf(hamming_prob)

    best_solution, best_value = sample_for_solution(G_func, nodes, G_max, shots, hamming_prob, max_weight, weights, n_qubits, dtype)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


def maxcut_tfim_streaming(
    G_func,
    nodes,
    quality=None,
    shots=None,
    is_base_maxcut_gpu=True
):
    dtype = opencl_context.dtype
    epsilon = opencl_context.epsilon
    wgs = opencl_context.work_group_size
    n_qubits = len(nodes)

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], [])

        if n_qubits == 1:
            return "0", 0, (nodes, [])

        if n_qubits == 2:
            weight = G_func(nodes[0], nodes[1])
            if weight < 0.0:
                return "00", 0, (nodes, [])

            return "01", weight, ([nodes[0]], [nodes[1]])

    if quality is None:
        quality = 3

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    n_steps = n_qubits << quality
    grid_size = n_steps * n_qubits

    if (not is_base_maxcut_gpu) or (not IS_OPENCL_AVAILABLE):
        return cpu_footer(shots, quality, n_qubits, G_func, nodes, dtype, epsilon)

    J_eff, degrees, G_max = init_J_and_z(G_func, nodes, dtype)

    delta_t = 1.0 / n_steps
    tot_t = 2.0 * n_steps * delta_t
    h_mult = 2.0 / tot_t

    args = np.empty(3, dtype=dtype)
    args[0] = delta_t
    args[1] = tot_t
    args[2] = h_mult

    # Move to GPU
    mf = cl.mem_flags
    args_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args)
    J_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=J_eff)
    deg_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=degrees)
    theta_buf = cl.Buffer(opencl_context.ctx, mf.READ_WRITE, size=(n_qubits * 4))

    # Warp size is 32:
    group_size = min(wgs, n_qubits)
    global_size = ((n_qubits + group_size - 1) // group_size) * group_size

    opencl_context.init_theta_kernel(
        opencl_context.queue, (global_size,), (group_size,),
        args_buf, np.int32(n_qubits), J_buf, deg_buf, theta_buf
    )

    hamming_prob = init_thresholds(n_qubits, dtype)

    # Warp size is 32:
    group_size = min(wgs, n_qubits - 1)
    grid_dim = n_steps * n_qubits * group_size

    # Move to GPU
    ham_buf = cl.Buffer(opencl_context.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hamming_prob)

    # Kernel execution
    opencl_context.maxcut_hamming_cdf_kernel(
        opencl_context.queue, (grid_dim,), (group_size,),
        np.int32(n_qubits), deg_buf, args_buf, J_buf, theta_buf, ham_buf
    )

    # Fetch results
    cl.enqueue_copy(opencl_context.queue, hamming_prob, ham_buf)
    opencl_context.queue.finish()

    args_buf.release()
    J_buf.release()
    deg_buf.release()
    theta_buf.release()

    args_buf = None
    J_buf = None
    deg_buf = None
    theta_buf = None

    max_weight = degrees.sum()
    degrees = None
    J_eff = 1.0 / (1.0 + epsilon - J_eff)

    return gpu_footer(shots, n_qubits, G_func, nodes, G_max, J_eff, degrees, hamming_prob, max_weight, dtype)
