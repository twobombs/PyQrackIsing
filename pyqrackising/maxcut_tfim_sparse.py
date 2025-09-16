import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange


IS_OPENCL_AVAILABLE = True
ctx = None
queue = None
try:
    import pyopencl as cl
    
    # Pick a device (GPU if available)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Load and build OpenCL kernels
    kernel_src = open(os.path.dirname(os.path.abspath(__file__)) + "/kernels.cl").read()
    program = cl.Program(ctx, kernel_src).build()
    maxcut_hamming_cdf_kernel = program.maxcut_hamming_cdf
except ImportError:
    IS_OPENCL_AVAILABLE = False


@njit
def probability_by_hamming_weight(J, h, z, theta, t, n_qubits):
    bias = np.empty(n_qubits - 1, dtype=np.float64)

    # critical angle
    theta_c = np.arcsin(max(-1.0, min(1.0, abs(h) / (z * J))))

    p = (
        pow(2.0, abs(J / h) - 1.0)
        * (1.0 + np.sin(theta - theta_c) * np.cos(1.5 * np.pi * J * t + theta) / (1.0 + np.sqrt(t)))
        - 0.5
    )

    if (p * n_qubits) >= 1024:
        return bias

    tot_n = 1.0 + 1.0 / pow(2.0, p * n_qubits)
    factor = pow(2.0, -p)
    n = 1.0
    for q in range(1, n_qubits):
        n *= factor
        bias[q - 1] = n
        tot_n += n
    bias /= tot_n

    if J > 0.0:
        return bias[::-1]

    return bias


@njit(parallel=True)
def maxcut_hamming_cdf(n_qubits, J_func, degrees, quality, hamming_prob):
    if n_qubits < 2:
        hamming_prob.fill(0.0)
        return

    n_steps = 1 << quality
    delta_t = 1.0 / n_steps
    tot_t = 2.0 * n_steps * delta_t
    h_mult = 2.0 / tot_t
    n_bias = n_qubits - 1

    theta = np.empty(n_qubits, dtype=np.float64)
    for q in range(n_qubits):
        J = J_func[q]
        z = degrees[q]
        theta[q] = np.arcsin(
            max(
                -1.0,
                min(
                    1.0,
                    (1.0 if J > 0.0 else -1.0) if np.isclose(abs(z * J), 0.0) else (abs(h_mult) / (z * J)),
                ),
            )
        )

    for qc in prange(n_qubits, n_steps * n_qubits):
        step = qc // n_qubits
        q = qc % n_qubits
        J_eff = J_func[q]
        if np.isclose(abs(J_eff), 0.0):
            continue
        z = degrees[q]
        theta_eff = theta[q]
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        h_t = h_mult * (tot_t - t)
        bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, t, n_qubits)
        last_bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, tm1, n_qubits)
        for i in range(n_bias):
            hamming_prob[i] += bias[i] - last_bias[i]

    tot_prob = hamming_prob.sum()
    hamming_prob /= tot_prob

    tot_prob = 0.0
    for i in range(n_bias):
        tot_prob += hamming_prob[i]
        hamming_prob[i] = tot_prob
    hamming_prob[-1] = 2.0


# Written by Elara (OpenAI custom GPT) and improved by Dan Strano
@njit
def local_repulsion_choice(adjacency_data, adjacency_rows, weights, n, m):
    """

    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency_data, adjacency_rows: CSR-format sparse adjacency data
    weights: float64 array of shape (n,)
    """

    weights = weights.copy()
    chosen = np.zeros(m, dtype=np.int32)   # store chosen indices
    available = np.ones(n, dtype=np.bool_) # True = available, False = not
    mask = np.zeros(n, dtype=np.bool_)
    chosen_count = 0

    for _ in range(m):
        # Count available
        total_w = 0.0
        for i in range(n):
            if available[i]:
                total_w += weights[i]
        if total_w <= 0:
            break

        # Normalize & sample
        r = np.random.rand()
        cum = 0.0
        node = -1
        for i in range(n):
            if available[i]:
                cum += weights[i] / total_w
                if r < cum:
                    node = i
                    break

        if node == -1:
            continue

        # Select node
        chosen[chosen_count] = node
        chosen_count += 1
        available[node] = False
        mask[node] = True

        # Repulsion: penalize neighbors
        for j in range(adjacency_rows[node], adjacency_rows[node + 1]):
            nbr = adjacency_data[j]
            if available[nbr]:
                weights[nbr] *= 0.5  # tunable penalty factor

    return mask


@njit
def compute_energy(sample, G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    energy = 0
    for u in range(n_qubits):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            if v < (u + 1):
                continue
            energy += G_data[col] * (1 if sample[u] == sample[v] else -1)

    return energy


@njit(parallel=True)
def compute_adjacency(G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    adjacency_rows = np.empty(n_qubits + 1, dtype=np.int32)
    adjacency_rows[0] = 0
    adjacency_data = np.full(G_data.shape[0], -1, dtype=np.int32)
    k = 0
    for i in prange(n_qubits):
        for j in range(G_rows[i], G_rows[i + 1]):
            if G_data[j] > 0.0:
                adjacency_data[k] = G_cols[j]
                k += 1
        adjacency_rows[i + 1] = k

    np.resize(adjacency_data, k)

    return adjacency_data, adjacency_rows


@njit(parallel=True)
def sample_for_solution(G_data, G_rows, G_cols, shots, thresholds, J_eff):
    n = G_rows.shape[0] - 1
    adjacency_data, adjacency_rows = compute_adjacency(G_data, G_rows, G_cols)
    weights = 1.0 / (1.0 + (2 ** -52) - J_eff)

    solutions = np.empty((shots, n), dtype=np.bool_)
    energies = np.empty(shots, dtype=np.float64)

    for s in prange(shots):
        # First dimension: Hamming weight
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        m += 1

        # Second dimension: permutation within Hamming weight
        sample = local_repulsion_choice(adjacency_data, adjacency_rows, weights, n, m)
        solutions[s] = sample
        energies[s] = compute_energy(sample, G_data, G_rows, G_cols)

    best_solution = solutions[np.argmin(energies)]

    best_value = 0.0
    for u in range(n):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            if v < (u + 1):
                continue
            if best_solution[u] != best_solution[v]:
                best_value += G_data[col]

    return best_solution, best_value


@njit
def init_J_and_z(G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float64)
    J_max = -float("inf")
    for n in range(n_qubits):
        start = G_rows[n]
        end = G_rows[n + 1]
        degree = end - start
        J = -np.mean(G_data[start:end]) / degree if degree > 0 else 0
        degrees[n] = degree
        J_eff[n] = J
        J_abs = abs(J)
        if J_abs > J_max:
            J_max = J_abs
    J_eff /= J_max

    return J_eff, degrees

@njit
def init_thresholds(n_qubits):
    n_bias = n_qubits - 1
    thresholds = np.empty(n_bias, dtype=np.float64)
    tot_prob = 0
    p = 1.0
    if n_qubits & 1:
        q = n_qubits // 2
        thresholds[q - 1] = p
        tot_prob = p
        p /= 2
    for q in range(1, n_qubits // 2):
        thresholds[q - 1] = p
        thresholds[n_bias - q] = p
        tot_prob += 2 * p
        p /= 2
    thresholds /= tot_prob

    return thresholds


@njit
def init_theta(delta_t, tot_t, h_mult, n_qubits, J_eff, degrees):
    theta = np.empty(n_qubits, dtype=np.float64)
    for q in range(n_qubits):
        J = J_eff[q]
        z = degrees[q]
        theta[q] = np.arcsin(
            max(
                -1.0,
                min(
                    1.0,
                    (1.0 if J > 0.0 else -1.0) if np.isclose(abs(z * J), 0.0) else (abs(h_mult) / (z * J)),
                ),
            )
        )

    return theta


def maxcut_tfim_sparse(
    G,
    quality=None,
    shots=None,
):
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_scipy_sparse_array(G, weight='weight')
    else:
        n_qubits = G.shape[0]
        nodes = list(range(n_qubits))
        G_m = G

    if n_qubits == 0:
        return "", 0, ([], [])

    if n_qubits == 1:
        return "0", 0, (nodes, [])

    if n_qubits == 2:
        weight = G_m[0, 1]
        if weight < 0.0:
            return "00", 0, (nodes, [])

        return "01", weight, ([nodes[0]], [nodes[1]])

    if quality is None:
        quality = 8

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    n_steps = 1 << quality
    grid_size = n_steps * n_qubits

    J_eff, degrees = init_J_and_z(G_m.data, G_m.indptr, G_m.indices)
    hamming_prob = init_thresholds(n_qubits)

    if IS_OPENCL_AVAILABLE and grid_size >= 128:
        delta_t = 1.0 / n_steps
        tot_t = 2.0 * n_steps * delta_t
        h_mult = 2.0 / tot_t
        theta = init_theta(delta_t, tot_t, h_mult, n_qubits, J_eff, degrees)
        args = np.empty(3, dtype=np.float32)
        args[0] = delta_t
        args[1] = tot_t
        args[2] = h_mult

        # Warp size is 32:
        group_size = n_qubits - 1
        if group_size > 256:
            group_size = 256
        grid_dim = n_steps * n_qubits * group_size

        # Move to GPU
        mf = cl.mem_flags
        args_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args)
        J_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=J_eff)
        deg_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=degrees)
        theta_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta)
        ham_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hamming_prob)

        # Kernel execution
        maxcut_hamming_cdf_kernel(
            queue, (grid_dim,), (group_size,),
            np.int32(n_qubits), deg_buf, args_buf, J_buf, theta_buf, ham_buf
        )

        # Fetch results
        cl.enqueue_copy(queue, hamming_prob, ham_buf)

        hamming_prob /= hamming_prob.sum()
        tot_prob = 0.0
        for i in range(n_qubits - 1):
            tot_prob += hamming_prob[i]
            hamming_prob[i] = tot_prob
        hamming_prob[-1] = 2.0
    else:
        maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, hamming_prob)

    best_solution, best_value = sample_for_solution(G_m.data, G_m.indptr, G_m.indices, shots, hamming_prob, J_eff)

    bit_string = ""
    l, r = [], []
    for i in range(len(best_solution)):
        if best_solution[i]:
            bit_string += "1"
            r.append(nodes[i])
        else:
            bit_string += "0"
            l.append(nodes[i])

    return bit_string, best_value, (l, r)
