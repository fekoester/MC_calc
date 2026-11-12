"""python -m mc_calc STATES INPUTS [options]: .npy or whitespace-separated text files."""

import argparse

import numpy as np

from . import information_processing_capacity, write_table


def _load(path):
    return np.load(path) if path.endswith(".npy") else np.loadtxt(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Information processing capacity of reservoir states.")
    ap.add_argument("states")
    ap.add_argument("inputs")
    ap.add_argument("--max-degree", type=int, default=3)
    ap.add_argument("--max-delay", type=int, default=50)
    ap.add_argument("--offset", type=int, default=None,
                    help="state row k belongs to input k + offset (default: aligned, same length)")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", default="memoryData", help="table in the C++ format")
    ap.add_argument("--plot", default="", help="save a figure (capacity per degree, linear memory function)")
    a = ap.parse_args(argv)
    r = information_processing_capacity(_load(a.states), _load(a.inputs), a.max_degree, a.max_delay, a.offset,
                                        ridge=a.ridge, threshold=a.threshold, patience=a.patience)
    write_table(r, a.out, a.max_degree)
    per = ", ".join(f"degree {d}: {c:.4f}" for d, c in enumerate(r.total_per_degree) if d > 0)
    print(f"{per} | total {r.total:.4f} of N = {r.n_features} (threshold {r.threshold:.2e})")
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r.plot(max_delay=a.max_delay)
        plt.tight_layout()
        plt.savefig(a.plot, dpi=130)
    return r


if __name__ == "__main__":
    main()
