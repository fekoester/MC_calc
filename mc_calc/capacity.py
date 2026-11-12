from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def legendre_table(u: np.ndarray, max_degree: int) -> np.ndarray:
    """P_n(u_t) for n = 0..max_degree (rows) and every input (columns), by the three-term recurrence."""
    u = np.asarray(u, dtype=float)
    P = np.empty((max_degree + 1, len(u)))
    P[0] = 1.0
    if max_degree >= 1:
        P[1] = u
    for n in range(2, max_degree + 1):
        P[n] = ((2.0 * n - 1.0) * u * P[n - 1] - (n - 1.0) * P[n - 2]) / n
    return P


def compositions(n: int, k: int) -> list[list[int]]:
    """Ordered k-tuples of positive integers summing to n (same order as getCombinations in Memory.h)."""
    if k == 1:
        return [[n]]
    if k == 2:
        return [[n - i - 1, i + 1] for i in range(n - 1)]
    out = []
    for i in range(1, n - (k - 1) + 1):
        out += [[i] + c for c in compositions(n - i, k - 1)]
    return out


def chi2_threshold(n_features: int, n_samples: int, probability: float = 1e-5, factor: float = 4.0) -> float:
    from scipy.stats import chi2

    return factor * chi2.isf(probability, n_features) / n_samples


@dataclass
class CapacityResult:
    """One entry per significant target, plus totals per degree (index d = total degree, index 0 unused)."""

    steps: list[tuple[int, ...]] = field(default_factory=list)
    degrees: list[tuple[int, ...]] = field(default_factory=list)
    capacities: list[float] = field(default_factory=list)
    total_per_degree: np.ndarray | None = None
    pure_per_degree: np.ndarray | None = None
    threshold: float = 0.0
    n_features: int = 0
    n_samples: int = 0
    cumulative: list[np.ndarray] = field(default_factory=list)  # running totals, as in the C++ output rows
    cumulative_pure: list[np.ndarray] = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(self.total_per_degree[1:].sum())

    def linear_memory_function(self, max_delay: int) -> np.ndarray:
        """Capacity of the linear targets P_1(u_{t-k}) for k = 0..max_delay-1 (0 where not significant)."""
        m = np.zeros(max_delay)
        for s, d, c in zip(self.steps, self.degrees, self.capacities):
            if d == (1,):
                m[s[0]] = c
        return m

    def plot(self, ax=None, max_delay: int | None = None):
        """Bar chart of the capacity per degree (left) and the linear memory function (right)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        degs = np.arange(1, len(self.total_per_degree))
        ax[0].bar(degs, self.total_per_degree[1:], color="tab:blue", label="all targets")
        ax[0].bar(degs, self.pure_per_degree[1:], color="tab:orange", alpha=0.8, label="single delay")
        ax[0].axhline(self.n_features, ls=":", color="gray", label=f"N = {self.n_features}")
        ax[0].set_xlabel("degree")
        ax[0].set_ylabel("capacity")
        ax[0].set_xticks(degs)
        ax[0].set_title(f"total capacity {self.total:.2f}")
        ax[0].legend(fontsize=8)
        D = max_delay or (max((s[0] for s in self.steps), default=0) + 1)
        ax[1].plot(np.arange(D), self.linear_memory_function(D), marker=".")
        ax[1].set_xlabel("delay k")
        ax[1].set_ylabel("capacity of $u_{t-k}$")
        ax[1].set_title("linear memory function")
        ax[1].set_ylim(-0.02, 1.02)
        return ax


def information_processing_capacity(
    states,
    inputs,
    max_degree: int = 3,
    max_delay: int = 50,
    offset: int | None = None,
    ridge: float = 1e-6,
    ridge_relative: bool = False,
    threshold: float | None = None,
    threshold_probability: float = 1e-5,
    threshold_factor: float = 4.0,
    use_threshold: bool = True,
    patience: int = 20,
) -> CapacityResult:
    """
    Information processing capacity of `states` (T x N) driven by `inputs` (uniform in [-1, 1]).

    Alignment: state row k belongs to input index k + offset.
      offset=None and len(inputs) == len(states): rows are aligned (inputs[t] drove states[t]); the first
        max_delay - 1 rows are dropped so that every delay is defined.
      offset=K (the C++ -start_target_row): len(inputs) must be >= T + K and K >= max_delay - 1.
    Arguments map to the C++ flags: max_degree (-max_power_nonlinear_memory), max_delay (-steps_for_memory),
    ridge / ridge_relative (-regularisation_factor[_relative]), threshold (-threshold), threshold_probability,
    threshold_factor, use_threshold=False (-do_without_threshold 1), patience (-patience; 0 = full scan).
    """
    X = np.array(states, dtype=float, copy=True)
    u = np.asarray(inputs, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("states must be a 2-D array (time x features)")
    if offset is None:
        if len(u) != len(X):
            raise ValueError("with offset=None, states and inputs must have the same length")
        offset = max_delay - 1
        X = X[offset:]
    T, N = X.shape
    if offset < max_delay - 1:
        raise ValueError(f"offset ({offset}) must be >= max_delay - 1 ({max_delay - 1})")
    if len(u) < T + offset:
        raise ValueError(f"need at least {T + offset} inputs for {T} state rows with offset {offset}, got {len(u)}")
    if np.abs(u).max() > 1 + 1e-9:
        raise ValueError("inputs must lie in [-1, 1] (Legendre polynomials are orthogonal for uniform inputs there)")

    if threshold is None:
        threshold = chi2_threshold(N, T, threshold_probability, threshold_factor)
    X -= X.mean(axis=0)
    G = X.T @ X
    lam = ridge * G.max() if ridge_relative else ridge
    G_inv = np.linalg.pinv(G + lam * np.eye(N))
    P = legendre_table(u, max_degree)
    rows = np.arange(T) + offset
    window = patience if patience > 0 else max_delay + 1

    res = CapacityResult(threshold=float(threshold), n_features=N, n_samples=T,
                         total_per_degree=np.zeros(max_degree + 1), pure_per_degree=np.zeros(max_degree + 1))

    def capacity(power_list, steps):
        y = np.ones(T)
        for d, s in zip(power_list, steps):
            y = y * P[d, rows - s]
        y = y - y.mean()
        p = X.T @ y
        return float(p @ G_inv @ p / (y @ y))

    for degree in range(1, max_degree + 1):
        for variables in range(1, degree + 1):
            for power_list in compositions(degree, variables):
                _scan(power_list, degree, max_delay, window, threshold, use_threshold, capacity, res)
    return res


def _scan(power_list, degree, max_delay, window, threshold, use_threshold, capacity, res):
    """Literal port of calcErrors / doLoopForOneLegendrePolynomial in Memory.h (including the early stop)."""
    L = len(power_list)
    st = {"low_last_layer": L, "low_last_quit": -1}
    steps = [0] * L
    start_steps = [0] * L
    acc_peak = [0] * L

    def loop(size, start_before):
        start = start_before + 1
        for i in range(start, max_delay):
            if size > 1:
                for j in range(size - 1, 0, -1):
                    acc_peak[L - j] = 0
                if st["low_last_quit"] >= size:
                    if st["low_last_quit"] == size:
                        st["low_last_quit"] = -1
                    break
                if i >= start + window + acc_peak[L - size] + 1:
                    st["low_last_layer"] = size
                steps[L - size] = i
                start_steps[L - size] = start
                loop(size - 1, i)
            else:
                if i >= start + window + 1:
                    st["low_last_layer"] = size
                steps[L - size] = i
                start_steps[L - size] = start
                c = capacity(power_list, steps)
                if c > threshold or not use_threshold:
                    for j in range(1, L + 1):
                        acc_peak[L - j] = steps[L - j] - start_steps[L - j]
                    res.total_per_degree[degree] += c
                    if L == 1:
                        res.pure_per_degree[degree] += c
                    res.steps.append(tuple(steps))
                    res.degrees.append(tuple(power_list))
                    res.capacities.append(c)
                    res.cumulative.append(res.total_per_degree[1:].copy())
                    res.cumulative_pure.append(res.pure_per_degree[1:].copy())
                elif i >= start + window + acc_peak[L - size]:
                    st["low_last_quit"] = st["low_last_layer"]
                    break

    loop(L, -1)


def write_table(res: CapacityResult, path: str, max_degree: int, precision: int = 6):
    """Write the same table as the C++ program (header + one row per significant target)."""
    P = max_degree
    header = ([f"step{j}" for j in range(1, P + 1)] + [f"degree{j}" for j in range(1, P + 1)] + ["capacity"]
              + [f"total_deg{j}" for j in range(1, P + 1)] + [f"pure_deg{j}" for j in range(1, P + 1)])
    fmt = lambda v: f"{v:.{precision}g}"  # noqa: E731
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for s, d, c, cum, pure in zip(res.steps, res.degrees, res.capacities, res.cumulative, res.cumulative_pure):
            vals = list(s) + [0] * (P - len(s)) + list(d) + [0] * (P - len(d))
            f.write(" ".join(str(v) for v in vals) + " " + fmt(c) + " " + " ".join(fmt(v) for v in cum) + " "
                    + " ".join(fmt(v) for v in pure) + "\n")
