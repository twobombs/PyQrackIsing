import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange

from .maxcut_tfim_util import binary_search, fix_cdf, get_cut, init_thresholds, maxcut_hamming_cdf, opencl_context, to_scipy_sparse_upper_triangular

IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


@njit
def update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node):
    # Select node
    used[node] = True

    # Repulsion: penalize neighbors
    for j in range(G_rows[node], G_rows[node + 1]):
        nbr = G_cols[j]
        if used[nbr]:
            continue
        weights[nbr] *= max(1.1920928955078125e-7, 1 - G_data[j] / max_edge)

    for nbr in range(node):
        if used[nbr]:
            continue
        start = G_rows[nbr]
        end = G_rows[nbr + 1]
        j = binary_search(G_cols[start:end], node) + start
        if j < end:
            weights[nbr] *= max(1.1920928955078125e-7, 1 - G_data[j] / max_edge)


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, tot_init_weight, n, m):
    """
    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency_data, adjacency_rows: CSR-format sparse adjacency data
    weights: float32 array of shape (n,)
    """

    weights = weights.copy()
    used = np.zeros(n, dtype=np.bool_) # False = available, True = used

    # First bit:
    r = np.random.rand()
    cum = 0.0
    node = 0
    for i in range(n):
        cum += weights[i]
        if (tot_init_weight * r) < cum:
            node = i
            break

    update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node)

    if m == 1:
        return used

    for _ in range(1, m - 1):
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

        # Update answer and weights
        update_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, n, used, node)

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

    used[node] = True

    return used


@njit
def compute_energy(sample, G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    energy = 0
    for u in range(n_qubits):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            val = G_data[col]
            energy += val if sample[u] == sample[v] else -val

    return energy


@njit(parallel=True)
def sample_for_solution(G_data, G_rows, G_cols, shots, thresholds, weights, dtype):
    shots = max(1, shots >> 1)
    n = G_rows.shape[0] - 1
    max_edge = G_data.max()
    tot_init_weight = weights.sum()

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
            sample = local_repulsion_choice(G_cols, G_data, G_rows, max_edge, weights, tot_init_weight, n, m)
            solutions[s] = sample
            energies[s] = compute_energy(sample, G_data, G_rows, G_cols)

        best_index = np.argmin(energies)
        energy = energies[best_index]
        if energy < best_energy:
            best_energy = energy
            best_solution = solutions[best_index].copy()
            improved = True

    best_value = 0.0
    for u in range(n):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            if best_solution[u] != best_solution[v]:
                best_value += G_data[col]

    return best_solution, best_value


@njit(parallel=True)
def init_J_and_z(G_data, G_rows, G_cols, dtype):
    n_qubits = G_rows.shape[0] - 1
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=dtype)
    for r in prange(n_qubits):
        # Row sum
        start = G_rows[r]
        end = G_rows[r + 1]
        degree = end - start
        val = G_data[start:end].sum()

        degrees[r] += degree
        J_eff[r] += val

        # Column sum
        for idx in range(start, end):
            c = G_cols[idx]
            degrees[c] += 1
            J_eff[c] += G_data[idx]

    J_max = -float("inf")
    for r in prange(n_qubits):
        J = J_eff[r]
        degree = degrees[r]
        J_eff[r] = -J / degree if degree > 0 else 0
        J_abs = abs(J)
        J_max = max(J_abs, J_max)
    J_eff /= J_max

    return J_eff, degrees


def run_sampling_opencl(G_m_csr, thresholds_np, shots, n, is_g_buf_reused):
    ctx = opencl_context.ctx
    queue = opencl_context.queue
    kernel = opencl_context.sample_for_solution_best_bitset_sparse_kernel
    dtype = opencl_context.dtype
    wgs = opencl_context.work_group_size

    local_size = min(wgs, shots)  # tune
    max_global_size = ((opencl_context.MAX_GPU_PROC_ELEM + local_size - 1) // local_size) * local_size  # corresponds to MAX_PROC_ELEM macro in OpenCL kernel program
    global_size = min(((shots + local_size - 1) // local_size) * local_size, max_global_size)
    num_groups = global_size // local_size

    # Bit-packing params
    words = (n + 31) // 32

    # Random seeds (host)
    rng_seeds_np = np.random.randint(1, 2**31-1, size=global_size, dtype=np.uint32)

    # Device buffers
    mf = cl.mem_flags
    G_data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m_csr.data)
    G_rows_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m_csr.indptr)
    G_cols_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m_csr.indices)
    thresholds_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thresholds_np)
    rng_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rng_seeds_np)

    # Solutions buffer: each work-item writes a candidate bitset
    solutions_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_groups * words * np.uint32().nbytes)

    # Best energies and solutions (per group)
    # We'll reuse solutions_buf to hold them after reduction (kernel copies winners into group slices)
    best_energies_np = np.empty(num_groups, dtype=dtype)
    best_energies_buf = cl.Buffer(ctx, mf.WRITE_ONLY, best_energies_np.nbytes)

    # Local memory buffers
    local_energy_buf = cl.LocalMemory(np.dtype(dtype).itemsize * local_size)
    local_index_buf = cl.LocalMemory(np.dtype(np.int32).itemsize * local_size)

    # Set kernel args
    kernel.set_args(
        G_data_buf,
        G_rows_buf,
        G_cols_buf,
        thresholds_buf,
        np.int32(n),
        np.int32(shots),
        dtype(G_m_csr.data.max()),
        rng_buf,
        solutions_buf,
        best_energies_buf,
        local_energy_buf,
        local_index_buf
    )

    # Launch
    cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))

    # Copy back
    cl.enqueue_copy(queue, best_energies_np, best_energies_buf)
    solutions_np = np.empty(num_groups * words, dtype=np.uint32)
    cl.enqueue_copy(queue, solutions_np, solutions_buf)  # all candidates, includes per-group winners
    queue.finish()

    # Host reduction
    best_group = np.argmax(best_energies_np)  # max cut
    best_energy = float(best_energies_np[best_group])
    best_solution_bits = solutions_np[best_group * words : (best_group + 1) * words]

    # Unpack bitset into boolean vector
    best_solution = np.zeros(n, dtype=np.bool_)
    for u in range(n):
        w = u >> 5
        b = u & 31
        best_solution[u] = (best_solution_bits[w] >> b) & 1

    if is_g_buf_reused:
        opencl_context.G_data_buf = G_data_buf
        opencl_context.G_rows_buf = G_rows_buf
        opencl_context.G_cols_buf = G_cols_buf

    return best_solution, best_energy


@njit
def cpu_footer(shots, quality, n_qubits, G_data, G_rows, G_cols, nodes, dtype, epsilon):
    J_eff, degrees = init_J_and_z(G_data, G_rows, G_cols, dtype)
    hamming_prob = init_thresholds(n_qubits, dtype)

    maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, hamming_prob, dtype)

    degrees = None
    J_eff = 1.0 / (1.0 + epsilon - J_eff)

    best_solution, best_value = sample_for_solution(G_data, G_rows, G_cols, shots, hamming_prob, J_eff, dtype)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


@njit
def gpu_footer(shots, n_qubits, G_data, G_rows, G_cols, weights, hamming_prob, nodes, dtype):
    fix_cdf(hamming_prob)

    best_solution, best_value = sample_for_solution(G_data, G_rows, G_cols, shots, hamming_prob, weights, dtype)

    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)


def maxcut_tfim_sparse(
    G,
    quality=None,
    shots=None,
    is_alt_gpu_sampling=False,
    is_g_buf_reused=False,
    is_base_maxcut_gpu=True
):
    dtype = opencl_context.dtype
    epsilon = opencl_context.epsilon
    wgs = opencl_context.work_group_size
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = to_scipy_sparse_upper_triangular(G, nodes, n_qubits, dtype)
    else:
        n_qubits = G.shape[0]
        nodes = list(range(n_qubits))
        G_m = G

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], [])

        if n_qubits == 1:
            return "0", 0, (nodes, [])

        if n_qubits == 2:
            weight = G_m[0, 1]
            if weight < 0.0:
                return "00", 0, (nodes, [])

            return "01", weight, ([nodes[0]], [nodes[1]])

    is_segmented = (G_m.data.nbytes << 1) > opencl_context.max_alloc
    if is_segmented and is_alt_gpu_sampling:
        print("[WARN] Using segmented solver, so disabling is_alt_gpu_sampling.")
        is_alt_gpu_sampling = False

    if quality is None:
        quality = 2

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    n_steps = n_qubits << quality
    grid_size = n_steps * n_qubits

    if (not is_base_maxcut_gpu) or (not IS_OPENCL_AVAILABLE):
        return cpu_footer(shots, quality, n_qubits, G_m.data, G_m.indptr, G_m.indices, nodes, dtype, epsilon)

    J_eff, degrees = init_J_and_z(G_m.data, G_m.indptr, G_m.indices, dtype)

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

    if not is_alt_gpu_sampling:
        degrees = None
        J_eff = 1.0 / (1.0 + epsilon - J_eff)

        return gpu_footer(shots, n_qubits, G_m.data, G_m.indptr, G_m.indices, J_eff, hamming_prob, nodes, dtype)

    fix_cdf(hamming_prob)
    best_solution, best_value = run_sampling_opencl(G_m, hamming_prob, shots, n_qubits, is_g_buf_reused)
    bit_string, l, r = get_cut(best_solution, nodes)

    return bit_string, best_value, (l, r)
