"""
Capacity of a small echo state network: an end-to-end example.

    python examples/esn_example.py            # prints the capacity per degree, writes docs/esn_capacity.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mc_calc import information_processing_capacity  # noqa: E402


def esn_states(u, n=50, rho=0.95, input_scale=0.5, bias=0.2, seed=0):
    rng = np.random.default_rng(seed)
    w_in = rng.uniform(-1, 1, n)
    w = rng.normal(size=(n, n))
    w *= rho / np.abs(np.linalg.eigvals(w)).max()
    b = bias * rng.uniform(-1, 1, n)
    x, out = np.zeros(n), np.empty((len(u), n))
    for t, ut in enumerate(u):
        x = np.tanh(w @ x + input_scale * w_in * ut + b)
        out[t] = x
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    washout, T = 500, 10_000
    u = rng.uniform(-1, 1, washout + T)  # inputs must be uniform in [-1, 1]
    X = esn_states(u)[washout:]
    u = u[washout:]
    res = information_processing_capacity(X, u, max_degree=4, max_delay=40)  # aligned: inputs[t] drove X[t]
    for d in range(1, 5):
        print(f"degree {d}: {res.total_per_degree[d]:.3f}")
    print(f"total {res.total:.3f} of N = {res.n_features}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res.plot(max_delay=40)
    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/esn_capacity.png", dpi=130)
