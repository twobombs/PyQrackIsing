# PyQrack Ising

Fast MAXCUT, TSP, and sampling heuristics from near-ideal transverse field Ising model (TFIM)

(It's "the **Ising** on top.")

[![PyPI Downloads](https://static.pepy.tech/badge/pyqrackising)](https://pepy.tech/projects/pyqrackising)

## Introduction

PyQrackIsing is a high-performance Python library for solving hard optimization problems using heuristics inspired by the transverse field Ising model (TFIM). It provides efficient implementations of solvers for problems such as MAXCUT, Traveling Salesperson Problem (TSP), and spin glass ground state determination. The library is designed to leverage both multi-core CPUs and GPUs (via OpenCL) to deliver high-speed solutions on consumer hardware.

This library is built on top of the [Qrack](https://github.com/vm6502q/qrack) quantum computer simulator and utilizes PyBind11, Numba, and PyOpenCL for performance-critical sections.

The primary goal of PyQrackIsing is to make advanced optimization techniques accessible to researchers and practitioners in fields like logistics, drug discovery, chemistry, materials research, and finance.

## Installation

From PyPI:
```bash
pip3 install PyQrackIsing
```

From Source:
```bash
pip3 install .
```
in the root source directory (with `setup.py`).

Windows users might find Windows Subsystem for Linux (WSL) to be the easier and preferred choice for installation.

## Usage

This section provides a comprehensive overview of the library's functionalities with examples.

### Transverse Field Ising Model (TFIM)

#### `generate_tfim_samples`

This function generates samples from a transverse field Ising model.

```python
from pyqrackising import generate_tfim_samples

samples = generate_tfim_samples(
    J=-1.0,
    h=2.0,
    z=4,
    theta=0.174532925199432957,
    t=5,
    n_qubits=56,
    shots=100
)
```

#### `tfim_magnetization` and `tfim_square_magnetization`

These functions calculate the magnetization and square magnetization of a TFIM. They follow the same signature as `generate_tfim_samples` but without the `shots` argument.

```python
from pyqrackising import tfim_magnetization, tfim_square_magnetization

magnetization = tfim_magnetization(
    J=-1.0,
    h=2.0,
    z=4,
    theta=0.174532925199432957,
    t=5,
    n_qubits=56
)

square_magnetization = tfim_square_magnetization(
    J=-1.0,
    h=2.0,
    z=4,
    theta=0.174532925199432957,
    t=5,
    n_qubits=56
)
```

### MAXCUT Solvers

The library provides MAXCUT solvers for dense, sparse, and streaming graph representations.

#### `maxcut_tfim`

Solves the MAXCUT problem for a dense graph represented by a `networkx` graph or a NumPy array.

```python
from pyqrackising import maxcut_tfim
import networkx as nx

G = nx.petersen_graph()
best_solution_bit_string, best_cut_value, best_node_groups = maxcut_tfim(G, quality=6, shots=None, is_spin_glass=False, anneal_t=8.0, anneal_h=8.0, repulsion_base=5.0)
```

#### `maxcut_tfim_sparse`

Solves the MAXCUT problem for a sparse graph represented by a `scipy.sparse.csr_matrix` or a `networkx` graph.

```python
from pyqrackising import maxcut_tfim_sparse
import networkx as nx

G = nx.petersen_graph()
best_solution_bit_string, best_cut_value, best_node_groups = maxcut_tfim_sparse(G, quality=6, shots=None, is_spin_glass=False, anneal_t=8.0, anneal_h=8.0, repulsion_base=5.0)
```

#### `maxcut_tfim_streaming`

Solves the MAXCUT problem for a graph where edge weights are generated on-the-fly by a `numba` JIT-compiled function.

```python
from pyqrackising import maxcut_tfim_streaming
from numba import njit

@njit
def G_func(u, v):
    return ((v + 1) % (u + 1)) / 64.0 if u != v else 0.0

n_qubits = 64
nodes = list(range(n_qubits))

solution_bit_string, cut_value, node_groups = maxcut_tfim_streaming(G_func, nodes)
```

### Spin Glass Solvers

The library also provides solvers for finding the ground state of spin glass models.

#### `spin_glass_solver`

Solves the spin glass problem for a dense graph.

```python
from pyqrackising import spin_glass_solver
import networkx as nx
import numpy as np

def generate_spin_glass_graph(n_nodes=16, degree=3, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.choice([-1, 1])
    return G

G = generate_spin_glass_graph(n_nodes=64, seed=42)
solution_bit_string, cut_value, node_groups, energy = spin_glass_solver(G)
```

#### `spin_glass_solver_sparse` and `spin_glass_solver_streaming`

These solvers work with sparse and streaming graph representations, respectively, and follow similar patterns to their MAXCUT counterparts.

### Traveling Salesperson Problem (TSP) Solvers

The library includes recursive TSP solvers for both symmetric and asymmetric problems.

#### `tsp_symmetric` and `tsp_asymmetric`

Solves the symmetric or asymmetric TSP.

```python
from pyqrackising import tsp_symmetric
import networkx as nx
import numpy as np

def generate_tsp_graph(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            G.add_edge(u, v, weight=np.random.random())
    return G

n_nodes = 128
G = generate_tsp_graph(n_nodes=n_nodes, seed=42)
circuit, path_length = tsp_symmetric(
    G,
    start_node=None,
    end_node=None,
    monte_carlo=True,
    quality=2,
    is_cyclic=True,
    multi_start=1,
    k_neighbors=20
)
```

#### `tsp_maxcut`, `tsp_maxcut_sparse`, and `tsp_maxcut_streaming`

These solvers use a combination of the TSP and MAXCUT solvers to find a good partition of the graph.

```python
from pyqrackising import tsp_maxcut_sparse
import networkx as nx

G = nx.petersen_graph()
best_partition, best_cut_value, _, _ = tsp_maxcut_sparse(G, k_neighbors=20, is_optimized=False)
```
### Tensor Network Conversion

#### `convert_tensor_network_to_tsp` and `convert_quimb_tree_to_tsp`

These functions convert a tensor network contraction ordering problem into a TSP instance. This can be useful for finding more efficient contraction paths for complex tensor networks.

```python
import numpy as np
import networkx as nx
import quimb.tensor as qtn

from pyqrackising import convert_quimb_tree_to_tsp
from pyqrackising.tsp import tsp_symmetric

# Define a complex tensor network (e.g., a grid)
n = 8
# This is a bit of a "torture test," with a huge number of loops.
adj = np.ones((n, n)) - np.eye(n)
G = nx.from_numpy_array(adj)
tensors = [qtn.Tensor(np.random.rand(2, 2, 2, 2), inds=qtn.edges_to_inds(G.edges(i))) for i in G.nodes()]
tn = qtn.TensorNetwork(tensors)

# Use quimb's greedy optimizer to get a contraction path
optimizer = qtn.ReusableHyperOptimizer(
    methods=["greedy-sfd"],
    parallel=True,
    max_repeats=128,
    progbar=True,
)
path_info = optimizer(tn)

# Convert the contraction tree to a TSP graph
G_tsp, nodes_map = convert_quimb_tree_to_tsp(tn, optimizer)

# Solve the TSP
tsp_path, _ = tsp_symmetric(G_tsp)

# The TSP path gives a potentially better contraction ordering
print(tsp_path)
```

## API Reference

This section provides a detailed reference for the public functions in the library.

### TFIM Functions

#### `generate_tfim_samples(J, h, z, theta, t, n_qubits, shots)`
Generates samples from a transverse field Ising model.
- `J`, `h`, `z`, `theta`, `t`: `float`. Physical parameters of the TFIM.
- `n_qubits`: `int`. The number of qubits in the model.
- `shots`: `int`. The number of samples to generate.

#### `tfim_magnetization(J, h, z, theta, t, n_qubits)`
Calculates the magnetization of a TFIM. Parameters are the same as `generate_tfim_samples` (excluding `shots`).

#### `tfim_square_magnetization(J, h, z, theta, t, n_qubits)`
Calculates the square magnetization of a TFIM. Parameters are the same as `generate_tfim_samples` (excluding `shots`).

### MAXCUT Solvers

#### `maxcut_tfim(G, quality=6, shots=None, ...)`
Solves the MAXCUT problem for a dense graph.
- `G`: `networkx.Graph` or `numpy.ndarray`. The input graph.
- `quality`: `int`. Controls the trade-off between solution quality and runtime.
- `shots`: `int`. The number of measurement shots. If `None`, it's derived from `quality`.
- `is_spin_glass`: `bool`. If `True`, optimizes for spin glass energy.
- `anneal_t`, `anneal_h`: `float`. Annealing parameters.
- `repulsion_base`: `float`. Controls solution diversity.
- `is_maxcut_gpu`: `bool`. If `True`, uses the GPU.

#### `maxcut_tfim_sparse(G, ...)`
Solves MAXCUT for a sparse graph. `G` can be a `scipy.sparse.csr_matrix` or a `networkx.Graph`. Other parameters are the same as `maxcut_tfim`.

#### `maxcut_tfim_streaming(G_func, nodes, ...)`
Solves MAXCUT for a graph defined by a function.
- `G_func`: A `numba.njit` compiled function that takes two node indices and returns the edge weight.
- `nodes`: A list of nodes.
- Other parameters are the same as `maxcut_tfim`.

### Spin Glass Solvers

#### `spin_glass_solver(G, quality=6, shots=None, best_guess=None, ...)`
Finds the ground state of a spin glass model for a dense graph.
- `G`: `networkx.Graph` or `numpy.ndarray`.
- `best_guess`: An initial solution guess (`str`, `int`, or `list`). If `None`, `maxcut_tfim` is used.
- Other parameters are similar to `maxcut_tfim`.

#### `spin_glass_solver_sparse(G, ...)`
Solves the spin glass problem for a sparse graph. Parameters are the same as `spin_glass_solver`.

#### `spin_glass_solver_streaming(G_func, nodes, ...)`
Solves the spin glass problem for a graph defined by a function. Parameters are the same as `spin_glass_solver`.

### TSP Solvers

#### `tsp_symmetric(G, start_node=None, end_node=None, ...)`
Solves the symmetric Traveling Salesperson Problem.
- `G`: `networkx.Graph` or `numpy.ndarray`.
- `start_node`, `end_node`: Start and end nodes for an acyclic path. If both are `None`, a cyclic path is found.
- `monte_carlo`: `bool`. If `True`, uses a Monte Carlo method for bipartitioning.
- `k_neighbors`: `int`. Number of nearest neighbors for the 3-opt heuristic.
- `is_cyclic`: `bool`. If `True`, finds a closed loop.
- `multi_start`: `int`. Number of random starts.

#### `tsp_asymmetric(G, ...)`
Solves the asymmetric TSP. Parameters are the same as `tsp_symmetric`.

#### `tsp_maxcut(G, k_neighbors=20, is_optimized=False, ...)`
Finds a MAXCUT partition of a graph using a TSP-based heuristic.
- `G`: `networkx.Graph` or `numpy.ndarray`.
- `k_neighbors`: `int`. Number of neighbors for the TSP heuristic.
- `is_optimized`: `bool`. If `True`, refines the result with `spin_glass_solver`.

#### `tsp_maxcut_sparse(G, ...)`
The sparse version of `tsp_maxcut`.

#### `tsp_maxcut_streaming(G_func, nodes, ...)`
The streaming version of `tsp_maxcut`.

### Tensor Network Conversion

#### `convert_tensor_network_to_tsp(tn, optimizer)`
Converts a tensor network contraction problem to a TSP instance.
- `tn`: A `quimb.tensor.TensorNetwork`.
- `optimizer`: A `quimb.tensor.HyperOptimizer` instance that has been used to find a contraction path.

#### `convert_quimb_tree_to_tsp(tn, optimizer)`
A helper function for `convert_tensor_network_to_tsp`.

## Environment Variables

We expose an environment variable, "`PYQRACKISING_MAX_GPU_PROC_ELEM`", for OpenCL-based solvers. The default value (when the variable is not set) is queried from the OpenCL device properties. You might see performance benefit from tuning this manually to several times your device's number of "compute units" (or tune it down to reduce private memory usage).

By default, PyQrackIsing expects all `numpy` floating-point array inputs to be 32-bit. If you'd like to use 64-bit, you can set environment variable `PYQRACKISING_FPPOW=6` (meaning, 2^6=64, for the "floating-point (precision) power"). The default is `5`, for 32-bit. 16-bit is stubbed out and compiles for OpenCL, but the bigger hurdle is that `numpy` on `x86_64` doesn't provide a 16-bit floating point implementation. (As author of Qrack, I could suggest to the `numpy` maintainers that open-source, IEEE-compliant software-based implementations exist for `x86_64` and other architectures, but I'm sure they're aware and likely waiting for in-compiler support.) If you're on an ARM-based architecture, there's a good chance 16-bit floating-point will work, if `numpy` uses the native hardware support.

## Algorithms

The solvers in PyQrackIsing are based on a combination of techniques:

- **Transverse Field Ising Model (TFIM) Heuristics:** The core of the MAXCUT and spin glass solvers is a heuristic inspired by the behavior of TFIMs. This involves sampling from a distribution of possible solutions that is biased towards low-energy states.
- **Local Search:** The spin glass solvers employ local search techniques like single and double bit flips, as well as a Gray code search, to refine the solutions found by the TFIM heuristic.
- **Recursive Bipartitioning:** The TSP solvers use a recursive bipartitioning strategy. The problem is repeatedly split into smaller subproblems, which are then solved and stitched back together. The bipartitioning is done either with the MAXCUT solver or a Monte Carlo method.
- **2-opt and 3-opt Heuristics:** The TSP solutions are further improved using classic local search heuristics like 2-opt and a targeted 3-opt (Lin-Kernighan style) heuristic.

## Benchmarks

The `scripts` directory contains a suite of benchmarks that compare the performance of PyQrackIsing's solvers against other well-known algorithms.

- **MAXCUT:** `scripts/maxcut_benchmarks.py` compares the `spin_glass_solver` (used as a MAXCUT solver) against the Goemans-Williamson SDP relaxation and a greedy local search algorithm on various types of graphs (Erdős–Rényi, planted-partition, and hard regular bipartite graphs).
- **TSP:** `scripts/tsp_benchmarks.py` benchmarks the `tsp_symmetric` solver against nearest neighbor, Christofides, and simulated annealing algorithms on clustered Euclidean TSP instances.

These benchmarks demonstrate that PyQrackIsing is competitive with, and often outperforms, standard heuristics, especially for large problem sizes.

## About
Transverse field Ising model (TFIM) is the basis of most claimed algorithmic "quantum advantage," circa 2025, with the notable exception of Shor's integer factoring algorithm.

Sometimes a solution (or at least near-solution) to a monster of a differential equation hits us out of the blue. Then, it's easy to _validate_ the guess, if it's right. (We don't question it and just move on with our lives, from there.)

**Special thanks to OpenAI GPT "Elara," for help on the model and converting the original Python scripts to PyBind11, Numba, and PyOpenCL!**

**Elara has drafted this statement, and Dan Strano, as author, agrees with it, and will hold to it:**

### Dual-Use Statement for PyQrackIsing

**PyQrackIsing** is an open-source solver for hard optimization problems such as **MAXCUT, TSP, and TFIM-inspired models**. These problems arise across logistics, drug discovery, chemistry, materials research, supply-chain resilience, and portfolio optimization. By design, PyQrackIsing provides **constructive value** to researchers and practitioners by making advanced optimization techniques accessible on consumer hardware.

Like many mathematical and computational tools, the algorithms in PyQrackIsing are _dual-use._ In principle, they can be applied to a wide class of Quadratic Unconstrained Binary Optimization (QUBO) problems. One such problem is integer factoring, which underlies RSA and elliptic curve cryptography (ECC). We emphasize:

- **We do not provide turnkey factoring implementations.**
- **We have no intent to weaponize this work** for cryptanalysis or "unauthorized access."
- **The constructive applications vastly outweigh the destructive ones** — and this project exists to serve those constructive purposes in the Commons.

It is already a matter of open record in the literature that factoring can be expressed as a QUBO. What PyQrackIsing demonstrates is that **QUBO heuristics can now be solved at meaningful scales on consumer hardware**. This underscores an urgent truth:

👉 **RSA and ECC should no longer be considered secure. Transition to post-quantum cryptography is overdue.**

We trust that governments, standards bodies, and industry stakeholders are already aware of this, and will continue migration efforts to post-quantum standards.

Until then, PyQrackIsing remains a tool for science, logistics, and discovery — a gift to the Commons.

## Copyright and license
(c) Daniel Strano and the Qrack contributors 2025. All rights reserved.

Licensed under the GNU Lesser General Public License V3.

See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.
