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
    def cuda_probability_by_hamming_weight(q, J, h, z, theta, t, n_qubits):
        # critical angle
        theta_c = np.arcsin(max(-1.0, min(1.0, abs(h) / (z * J))))

        p = (
            pow(2.0, abs(J / h) - 1.0)
            * (1.0 + math.sin(theta - theta_c) * math.cos(1.5 * math.pi * J * t + theta) / (1.0 + math.sqrt(t)))
            - 0.5
        )

        if (p * (n_qubits + 2.0)) >= 1024.0:
            return 0.0

        result = (pow(2.0, (n_qubits + 2.0) * p) - 1.0) * pow(2.0, -((n_qubits + 1.0) * p) - p * q) / (pow(2.0, p) - 1.0)

        if math.isnan(result) or math.isinf(result):
            return 0.0

        return result

    @cuda.jit
    def cuda_maxcut_hamming_cdf_wide(delta_t, tot_t, h_mult, J_func, degrees, theta, hamming_prob):
        step = cuda.blockIdx.x
        qi = cuda.blockIdx.y
        J_eff = J_func[qi]
        z = degrees[qi]
        if abs(z * J_eff) <= (2 ** (-52)):
            return

        n_qubits = cuda.gridDim.y
        theta_eff = theta[qi]
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        h_t = h_mult * (tot_t - t)

        qo = cuda.threadIdx.x
        if J_eff > 0.0:
            qo = cuda.blockDim.x - (1 + qo)
        diff = cuda_probability_by_hamming_weight(qo, J_eff, h_t, z, theta_eff, t, n_qubits)
        diff -= cuda_probability_by_hamming_weight(qo, J_eff, h_t, z, theta_eff, tm1, n_qubits)
        hamming_prob[qo] += diff

    @cuda.jit
    def cuda_maxcut_hamming_cdf(n_qubits, delta_t, tot_t, h_mult, J_func, degrees, theta, hamming_prob):
        step = cuda.blockIdx.x // n_qubits
        qi = cuda.blockIdx.x % n_qubits
        J_eff = J_func[qi]
        z = degrees[qi]
        if abs(z * J_eff) <= (2 ** (-52)):
            return

        theta_eff = theta[qi]
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        h_t = h_mult * (tot_t - t)

        n_threads = cuda.gridDim.y
        qo = cuda.threadIdx.x
        while qo < n_qubits:
            _qo = (n_qubits - (1 + qo)) if J_eff > 0.0 else qo
            diff = cuda_probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, t, n_qubits)
            diff -= cuda_probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, tm1, n_qubits)
            hamming_prob[_qo] += diff
            qo += n_threads

except:
    IS_CUDA_AVAILABLE = False


@njit
def probability_by_hamming_weight(J, h, z, theta, t, n_qubits):
    bias = np.empty(n_qubits - 1, dtype=np.float64)

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


# Written by Elara (OpenAI custom GPT)
@njit
def local_repulsion_choice(adjacency, degrees, weights, n, m):
    """

    Pick m nodes out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    adjacency: 2D int array (n x max_deg), padded with -1

    degrees: int array of shape (n,)
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
        for j in range(degrees[node]):
            nbr = adjacency[node, j]
            if nbr < 0:
                break
            if available[nbr]:
                weights[nbr] *= 0.5  # tunable penalty factor

    return mask


@njit
def compute_energy(sample, G_m, n_qubits):
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            eigen = 1 if sample[u] == sample[v] else -1
            energy += G_m[u, v] * eigen

    return energy


@njit(parallel=True)
def sample_for_solution(G_m, shots, thresholds, degrees, J_eff, n):
    adjacency = compute_adjacency(G_m, degrees.max())
    weights = 1.0 / (1.0 + (2 ** -52) - J_eff)

    best_solution = np.zeros(n, dtype=np.bool_)
    best_energy = compute_energy(best_solution, G_m, n)

    for s in prange(shots):
        # First dimension: Hamming weight
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        m += 1

        # Second dimension: permutation within Hamming weight
        sample = local_repulsion_choice(adjacency, degrees, weights, n, m)

        energy = compute_energy(sample, G_m, n)
        if energy < best_energy:
            best_energy = energy
            best_solution = sample

    best_value = 0
    for u in range(n):
        for v in range(u + 1, n):
            if best_solution[u] != best_solution[v]:
                best_value += G_m[u, v]

    return best_solution, float(best_value)


@njit
def init_J_and_z(G_m):
    n_qubits = len(G_m)
    degrees = np.empty(n_qubits, dtype=np.uint32)
    J_eff = np.empty(n_qubits, dtype=np.float64)
    J_max = -float("inf")
    for n in range(n_qubits):
        degree = sum(G_m[n] != 0.0)
        J = -G_m[n].sum() / degree if degree > 0 else 0
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

@njit(parallel=True)
def compute_adjacency(G_m, max_degree):
    n_qubits = len(G_m)
    adjacency = np.full((n_qubits, max_degree), -1, dtype=np.int32)
    for i in prange(n_qubits):
        k = 0
        for j in range(n_qubits):
            if i == j:
                continue
            if G_m[i, j] > 0.0:
                adjacency[i, k] = j
                k += 1

    return adjacency


def maxcut_tfim(
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
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0)
    else:
        n_qubits = len(G)
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

    J_eff, degrees = init_J_and_z(G_m)
    thresholds = init_thresholds(n_qubits)

    if IS_CUDA_AVAILABLE and cuda.is_available() and grid_size >= 128:
        delta_t = 1.0 / n_steps
        tot_t = 2.0 * n_steps * delta_t
        h_mult = 2.0 / tot_t
        theta = init_theta(delta_t, tot_t, h_mult, n_qubits, J_eff, degrees)

        # Warp size is 32:
        group_size = n_qubits - 1
        grid_dims = (n_steps, n_qubits)
        if group_size <= 256:
            cuda_maxcut_hamming_cdf_wide[grid_dims, group_size](delta_t, tot_t, h_mult, J_eff, degrees, theta, thresholds)
        else:
            cuda_maxcut_hamming_cdf[n_steps * n_qubits, 128](n_qubits, delta_t, tot_t, h_mult, J_eff, degrees, theta, thresholds)

        thresholds /= thresholds.sum()
        tot_prob = 0.0
        for i in range(n_qubits - 1):
            tot_prob += thresholds[i]
            thresholds[i] = tot_prob
        thresholds[-1] = 2.0
    else:
        maxcut_hamming_cdf(n_qubits, J_eff, degrees, quality, thresholds)

    best_solution, best_value = sample_for_solution(G_m, shots, thresholds, degrees, J_eff, n_qubits)

    bit_string = "".join(["1" if b else "0" for b in best_solution])
    bit_list = list(bit_string)
    l, r = [], []
    for i in range(len(bit_list)):
        b = bit_list[i] == "1"
        if b:
            r.append(nodes[i])
        else:
            l.append(nodes[i])

    return bit_string, best_value, (l, r)
