from .maxcut_tfim import maxcut_tfim
from .maxcut_tfim_util import opencl_context
from .spin_glass_solver_util import get_cut_from_bit_array, int_to_bitstring
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import os


IS_OPENCL_AVAILABLE = True
try:
    import pyopencl as cl
except ImportError:
    IS_OPENCL_AVAILABLE = False


@njit
def evaluate_cut_edges(theta_bits, G_m):
    n_qubits = len(G_m)
    cut = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            if theta_bits[u] != theta_bits[v]:
                cut += G_m[u, v]

    return cut


@njit
def compute_energy(theta_bits, G_m):
    n_qubits = len(G_m)
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            energy += G_m[u, v] if theta_bits[u] == theta_bits[v] else -G_m[u, v]

    return energy


@njit
def bootstrap_worker(theta, G_m, indices):
    local_theta = theta.copy()
    for i in indices:
        local_theta[i] = not local_theta[i]
    energy = compute_energy(local_theta, G_m)

    return energy


@njit(parallel=True)
def bootstrap(best_theta, G_m, indices_array, k, min_energy, dtype):
    n = len(indices_array) // k
    energies = np.empty(n, dtype=dtype)
    for i in prange(n):
        j = i * k
        energies[i] = bootstrap_worker(best_theta, G_m, indices_array[j : j + k])

    energy = energies.min()
    if energy < min_energy:
        index_match = np.random.choice(np.where(energies == energy)[0])
        indices = indices_array[(index_match * k) : ((index_match + 1) * k)]
        min_energy = energy
        for i in indices:
            best_theta[i] = not best_theta[i]

    return min_energy


def run_bootstrap_opencl(best_theta, G_m_buf, indices_array_np, k, min_energy, is_segmented, segment_size):
    ctx = opencl_context.ctx
    queue = opencl_context.queue
    bootstrap_kernel = opencl_context.bootstrap_segmented_kernel if is_segmented else opencl_context.bootstrap_kernel
    dtype = opencl_context.dtype
    epsilon = opencl_context.epsilon
    wgs = opencl_context.work_group_size

    n = best_theta.shape[0]
    combo_count = len(indices_array_np) // k

    best_theta_np = np.array([(1 if b else 0) for b in best_theta], dtype=np.int8)

    # Args: [n, k, combo_count, segment_size]
    args_np = np.array([n, k, combo_count, segment_size], dtype=np.int32)

    # Buffers
    mf = cl.mem_flags
    best_theta_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=best_theta_np)
    indices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indices_array_np)
    args_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args_np)

    # Local memory allocation (1 float per work item)
    local_size = min(wgs, n)
    max_global_size = ((opencl_context.MAX_GPU_PROC_ELEM + local_size - 1) // local_size) * local_size  # corresponds to MAX_PROC_ELEM macro in OpenCL kernel program
    global_size = min(((combo_count + local_size - 1) // local_size) * local_size, max_global_size)
    local_energy_buf = cl.LocalMemory(np.dtype(dtype).itemsize * local_size)
    local_index_buf = cl.LocalMemory(np.dtype(np.int32).itemsize * local_size)

    # Allocate min_energy and min_index result buffers per workgroup
    min_energy_host = np.empty(global_size, dtype=dtype)
    min_index_host = np.empty(global_size, dtype=np.int32)

    min_energy_buf = cl.Buffer(ctx, mf.WRITE_ONLY, min_energy_host.nbytes)
    min_index_buf = cl.Buffer(ctx, mf.WRITE_ONLY, min_index_host.nbytes)

    # Set kernel args
    if is_segmented:
        bootstrap_kernel.set_args(
            np.random.randint(1<<32, dtype='uint32'),
            G_m_buf[0],
            G_m_buf[1],
            G_m_buf[2],
            G_m_buf[3],
            best_theta_buf,
            indices_buf,
            args_buf,
            min_energy_buf,
            min_index_buf,
            local_energy_buf,
            local_index_buf
        )
    else:
        bootstrap_kernel.set_args(
            np.random.randint(1<<32, dtype='uint32'),
            G_m_buf,
            best_theta_buf,
            indices_buf,
            args_buf,
            min_energy_buf,
            min_index_buf,
            local_energy_buf,
            local_index_buf
        )

    cl.enqueue_nd_range_kernel(queue, bootstrap_kernel, (global_size,), (local_size,))

    # Read results
    cl.enqueue_copy(queue, min_energy_host, min_energy_buf)
    cl.enqueue_copy(queue, min_index_host, min_index_buf)
    queue.finish()

    # Find global minimum
    energy = min_energy_host.min()
    if min_energy < energy:
        return min_energy

    atol = dtype(epsilon)
    rtol = dtype(0)
    choices = np.where(np.isclose(min_energy_host, energy, atol=atol, rtol=rtol))[0]
    best_i = np.random.choice(choices) if len(choices) else np.argmin(min_energy_host)

    flip_index_start = best_i * k
    indices_to_flip = indices_array_np[flip_index_start : flip_index_start + k]

    for i in indices_to_flip:
        best_theta[i] = not best_theta[i]

    return min_energy_host[best_i]


def spin_glass_solver(
    G,
    quality=None,
    shots=None,
    best_guess=None,
    is_combo_maxcut_gpu=True,
    is_spin_glass=True,
    anneal_t=None,
    anneal_h=None,
    repulsion_base=None
):
    dtype = opencl_context.dtype
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0, dtype=dtype)
    else:
        n_qubits = len(G)
        nodes = list(range(n_qubits))
        G_m = G

    segment_size = G_m.shape[0] ** 2
    is_segmented = (G_m.nbytes << 1) > opencl_context.max_alloc
    if is_segmented and is_alt_gpu_sampling:
        print("[WARN] Using segmented solver, so disabling is_alt_gpu_sampling.")
        is_alt_gpu_sampling = False

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], []), 0

        if n_qubits == 1:
            return "0", 0, (nodes, []), 0

        if n_qubits == 2:
            weight = G_m[0, 1]
            if weight < 0.0:
                return "00", 0, (nodes, []), weight

            return "01", weight, ([nodes[0]], [nodes[1]]), -weight

    bitstring = ""
    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        bitstring, _, _ = maxcut_tfim(G_m, quality=quality, shots=shots, is_spin_glass=is_spin_glass, anneal_t=anneal_t, anneal_h=anneal_h, repulsion_base=repulsion_base)
    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)

    if is_combo_maxcut_gpu and IS_OPENCL_AVAILABLE:
        if not (opencl_context.G_m_buf is None):
            G_m_buf = opencl_context.G_m_buf
        else:
            mf = cl.mem_flags
            ctx = opencl_context.ctx
            if is_segmented:
                o_shape = segment_size
                segment_size = (segment_size + 3) >> 2
                n_shape = segment_size << 2
                _G_m = np.reshape(G_m, (o_shape,))
                if n_shape != o_shape:
                    np.resize(_G_m, (n_shape,))
                _G_m_segments = np.split(_G_m, 4)
                G_m_buf = [
                    cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seg)
                    for seg in _G_m_segments
                ]
                _G_m = None
                _G_m_segments = None
            else:
                G_m_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m)

    min_energy = compute_energy(best_theta, G_m)
    improved = True
    correction_quality = 1
    combos_list = []
    while improved:
        improved = False
        k = 1
        while k <= correction_quality:
            if n_qubits < k:
                break

            combos = []
            if len(combos_list) < k:
                combos = np.array(list(
                    item for sublist in itertools.combinations(range(n_qubits), k) for item in sublist
                ))
                combos_list.append(combos)
            else:
                combos = combos_list[k - 1]

            if is_combo_maxcut_gpu and IS_OPENCL_AVAILABLE:
                energy = run_bootstrap_opencl(best_theta, G_m_buf, combos, k, min_energy, is_segmented, segment_size)
            else:
                energy = bootstrap(best_theta, G_m, combos, k, min_energy, dtype)

            if energy < min_energy:
                min_energy = energy
                improved = True
                if correction_quality < (k + 1):
                    correction_quality = k + 1
                break

            k = k + 1

    bitstring, l, r = get_cut_from_bit_array(best_theta, nodes)
    cut_value = evaluate_cut_edges(best_theta, G_m)
    opencl_context.G_m_buf = None

    return bitstring, float(cut_value), (l, r), float(min_energy)
