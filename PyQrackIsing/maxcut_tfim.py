import math
import networkx as nx
import numpy as np
from numba import njit, prange
IS_CUDA_AVAILABLE = True
try:
    from numba import cuda, types
    from numba.core.errors import NumbaPerformanceWarning
    import warnings

    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    @cuda.jit(device=True)
    def cuda_probability_by_hamming_weight(q, J, h, z, theta, t, n_qubits, bias):
        if J > 0.0:
            q = n_qubits - (2 + q)
    
        # critical angle
        theta_c = np.arcsin(max(-1.0, min(1.0, abs(h) / (z * J))))

        p = (
            pow(2.0, abs(J / h) - 1.0)
            * (1.0 + math.sin(theta - theta_c) * math.cos(1.5 * math.pi * J * t + theta) / (1.0 + math.sqrt(t)))
            - 0.5
        )

        if (p * n_qubits) >= 1024:
            bias[q] = 0.0
        else:
            norm = pow(2.0, -(n_qubits + 1) * p) * (pow(2.0, (n_qubits + 2) * p) - 1.0) / (pow(2, p) - 1.0)
            bias[q] = 1.0 / pow(2.0, p * q) / norm

    @cuda.jit
    def cuda_maxcut_hamming_cdf(delta_t, tot_t, h_mult, J_func, degrees, theta, hamming_prob):
        bias = cuda.shared.array(0, dtype=np.float64)
        step = cuda.blockIdx.x
        qi = cuda.blockIdx.y
        J_eff = J_func[qi]
        z = degrees[qi]
        if abs(z * J_eff) <= (2 ** (-54)):
            return

        n_qubits = cuda.gridDim.y
        theta_eff = theta[qi]
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        h_t = h_mult * (tot_t - t)

        qo = cuda.threadIdx.x
        cuda_probability_by_hamming_weight(qo, J_eff, h_t, z, theta_eff, t, n_qubits, bias)
        hamming_prob[qo] += bias[qo]
        cuda_probability_by_hamming_weight(qo, J_eff, h_t, z, theta_eff, tm1, n_qubits, bias)
        hamming_prob[qo] -= bias[qo]

except:
    IS_CUDA_AVAILABLE = False


@njit
def probability_by_hamming_weight(J, h, z, theta, t, n_qubits):
    bias = np.zeros(n_qubits - 1)

    # critical angle
    theta_c = np.arcsin(
        max(
            -1.0,
            min(
                1.0,
                (1.0 if J > 0.0 else -1.0) if np.isclose(abs(z * J), 0.0) else (abs(h) / (z * J)),
            ),
        )
    )

    p = (
        pow(2.0, abs(J / h) - 1.0)
        * (1.0 + np.sin(theta - theta_c) * np.cos(1.5 * np.pi * J * t + theta) / (1.0 + np.sqrt(t)))
        - 0.5
    )

    if (p * n_qubits) >= 1024:
        return bias

    tot_n = 1.0 + 1.0 / pow(2.0, p * n_qubits)
    for q in range(1, n_qubits):
        n = 1.0 / pow(2.0, p * q)
        bias[q - 1] = n
        tot_n += n
    bias /= tot_n

    if J > 0.0:
        return bias[::-1]

    return bias


@njit(parallel=True)
def maxcut_hamming_cdf(n_qubits, J_func, degrees, quality, hamming_prob):
    if n_qubits < 2:
        return np.empty(0, dtype=np.float64)

    n_steps = 1 << quality
    delta_t = 1.0 / n_steps
    tot_t = n_steps * delta_t
    h_mult = 32.0 / tot_t
    n_bias = n_qubits - 1

    theta = np.zeros(n_qubits)
    for q in prange(n_qubits):
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

    tot_prob = sum(hamming_prob)
    hamming_prob /= tot_prob

    tot_prob = 0.0
    for i in range(n_bias):
        tot_prob += hamming_prob[i]
        hamming_prob[i] = tot_prob
    hamming_prob[-1] = 1.0


# Written by Elara (OpenAI custom GPT)
def local_repulsion_choice(nodes, adjacency, degrees, weights, n, m):
    """
    Pick m nodes (bit positions) out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    """

    # Base weights: inverse degree
    # degrees = np.array([len(adjacency.get(i, [])) for i in range(n)], dtype=np.float64)
    # weights = 1.0 / (degrees + 1.0)
    weights = weights.copy()

    chosen = []
    available = set(range(len(nodes)))

    for _ in range(m):
        if not available:
            break

        # Normalize weights over remaining nodes
        sub_weights = np.array([weights[i] for i in available], dtype=np.float64)
        sub_weights /= sub_weights.sum()
        sub_nodes = list(available)

        # Sample one node
        idx = np.random.choice(len(sub_nodes), p=sub_weights)
        node = sub_nodes[idx]
        chosen.append(node)

        # Remove node from available
        available.remove(node)

        # Repulsion: penalize neighbors
        for nbr in adjacency.get(nodes[node], []):
            idx = nodes.index(nbr)
            if idx in available:
                weights[idx] *= 0.5  # halve neighbor's weight (tunable!)

    # Build integer mask
    mask = 0
    for pos in chosen:
        mask |= 1 << pos

    return mask


def evaluate_cut_edges(samples, edge_keys, edge_values):
    best_value = float("-inf")
    best_solution = None
    best_cut_edges = None

    for state in samples:
        cut_edges = []
        cut_value = 0
        for i in range(len(edge_values)):
            k = i << 1
            u, v = edge_keys[k], edge_keys[k + 1]
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_value += edge_values[i]

        if cut_value > best_value:
            best_value = cut_value
            best_solution = state

    return best_solution, float(best_value)


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def maxcut_tfim(
    G,
    quality=None,
    shots=None,
):
    # Number of qubits/nodes
    nodes = list(G.nodes())
    n_qubits = len(nodes)

    if n_qubits == 0:
        return "", 0, ([], [])

    if n_qubits == 1:
        return "0", 0, ([nodes[0]], [])

    # Warp size is 32:
    group_size = n_qubits - 1
    shared_size = (n_qubits - 1) * 8

    if quality is None:
        quality = 8

    if shots is None:
        # Number of measurement shots
        shots = n_qubits << quality

    n_steps = 1 << quality
    grid_size = n_steps * n_qubits
    grid_dims = (n_steps, n_qubits)

    J_eff = np.array(
        [
            -sum(edge_attributes.get("weight", 1.0) for _, edge_attributes in G.adj[n].items())
            for n in nodes
        ],
        dtype=np.float64,
    )
    degrees = np.array(
        [
            sum(abs(edge_attributes.get("weight", 1.0)) for _, edge_attributes in G.adj[n].items())
            for n in nodes
        ],
        dtype=np.float64,
    )
    # thresholds = tfim_sampler._maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality)
    n_bias = n_qubits - 1
    tot_prob = 2.0
    thresholds = np.zeros(n_bias)
    for q in range(1, n_qubits // 2):
        n = math.comb(n_qubits, q)
        thresholds[q - 1] = n
        thresholds[n_bias - q] = n
        tot_prob += n << 1
    if n_qubits & 1:
        q = n_qubits // 2
        n = math.comb(n_qubits, q)
        thresholds[q - 1] = n
        tot_prob += n
    thresholds /= tot_prob

    if IS_CUDA_AVAILABLE and cuda.is_available() and grid_size >= 128:
        delta_t = 1.0 / n_steps
        tot_t = n_steps * delta_t
        h_mult = 32.0 / tot_t

        theta = np.zeros(n_qubits)
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

        cuda_maxcut_hamming_cdf[grid_dims, group_size, 0, shared_size](delta_t, tot_t, h_mult, J_eff, degrees, theta, thresholds)

        tot_prob = sum(thresholds)
        thresholds /= tot_prob

        tot_prob = 0.0
        for i in range(n_bias):
            tot_prob += thresholds[i]
            thresholds[i] = tot_prob
        thresholds[-1] = 1.0
    else:
        maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, thresholds)
    G_dict = nx.to_dict_of_lists(G)
    J_max = max(J_eff)
    weights = 1.0 / (1.0 + (J_max - J_eff))
    samples = []
    for s in range(shots):
        # First dimension: Hamming weight
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        m += 1
        # Second dimension: permutation within Hamming weight
        samples.append(local_repulsion_choice(nodes, G_dict, degrees, weights, n_qubits, m))

    # We only need unique instances
    samples = list(set(samples))

    edge_keys = []
    edge_values = []
    for u, v, data in G.edges(data=True):
        edge_keys.append(nodes.index(u))
        edge_keys.append(nodes.index(v))
        edge_values.append(data.get("weight", 1.0))

    best_solution, best_value = evaluate_cut_edges(samples, edge_keys, edge_values)

    bit_string = int_to_bitstring(best_solution, n_qubits)
    bit_list = list(bit_string)
    l, r = [], []
    for i in range(len(bit_list)):
        b = bit_list[i] == "1"
        if b:
            r.append(nodes[i])
        else:
            l.append(nodes[i])

    return bit_string, best_value, (l, r)
