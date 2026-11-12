import os
import shutil
import subprocess

import numpy as np
import pytest

from mc_calc import information_processing_capacity, legendre_table, write_table
from mc_calc.capacity import compositions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(0)
T, W = 4000, 100
U = RNG.uniform(-1, 1, T + W)


def delayed(d):  # state row k <-> input k + W
    return U[W - d: W - d + T]


def esn(n=30, rho=0.9, bias=0.0, seed=1):
    r = np.random.default_rng(seed)
    w_in, w = r.uniform(-1, 1, n), r.normal(size=(n, n))
    w *= rho / np.abs(np.linalg.eigvals(w)).max()
    x, out = np.zeros(n), []
    for t in range(T + W):
        x = np.tanh(w @ x + 0.5 * w_in * U[t] + bias)
        out.append(x.copy())
    return np.array(out)[W:]


def test_legendre_polynomials():
    x = np.linspace(-1, 1, 7)
    P = legendre_table(x, 3)
    np.testing.assert_allclose(P[2], (3 * x**2 - 1) / 2)
    np.testing.assert_allclose(P[3], (5 * x**3 - 3 * x) / 2)


def test_compositions_cover_every_ordered_split():
    assert compositions(3, 2) == [[2, 1], [1, 2]]
    assert sorted(map(tuple, compositions(4, 3))) == [(1, 1, 2), (1, 2, 1), (2, 1, 1)]


def test_delay_line_has_exactly_linear_capacity_n():
    X = np.stack([delayed(d) for d in range(10)], 1)
    r = information_processing_capacity(X, U, offset=W)
    np.testing.assert_allclose(r.total_per_degree[1:], [10, 0, 0], atol=1e-6)
    np.testing.assert_allclose(r.linear_memory_function(12), [1] * 10 + [0, 0], atol=1e-6)


def test_quadratic_features_give_degree_two_capacity():
    X = np.stack([delayed(d) for d in range(5)] + [delayed(0) ** 2, delayed(0) * delayed(3)], 1)
    r = information_processing_capacity(X, U, offset=W)
    np.testing.assert_allclose(r.total_per_degree[1:], [5, 2, 0], atol=1e-6)
    assert ((0,), (2,)) in [(s, d) for s, d in zip(r.steps, r.degrees)]
    assert ((0, 3), (1, 1)) in [(s, d) for s, d in zip(r.steps, r.degrees)]


def test_odd_reservoir_has_no_even_capacity_and_total_is_bounded():
    X = esn()
    r = information_processing_capacity(X, U, offset=W, max_degree=4)
    assert r.total_per_degree[2] == 0 and r.total_per_degree[4] == 0  # tanh without bias is odd
    assert 0 < r.total <= X.shape[1] + 1e-6
    rb = information_processing_capacity(esn(bias=0.3), U, offset=W)  # a bias breaks the symmetry
    assert rb.total_per_degree[2] > 0.1 and rb.total <= 30 + 1e-6


def test_aligned_inputs_and_errors():
    X = np.stack([U[W:][:T]], 1)
    r = information_processing_capacity(np.stack([U, np.roll(U, 1)], 1), U, max_delay=5)
    assert r.total_per_degree[1] == pytest.approx(2, abs=1e-3)
    with pytest.raises(ValueError):
        information_processing_capacity(X, U[:T], offset=W)  # too few inputs
    with pytest.raises(ValueError):
        information_processing_capacity(X, U, offset=3)  # delays would reach before the first input
    with pytest.raises(ValueError):
        information_processing_capacity(X, 2 * U, offset=W)  # inputs outside [-1, 1]


def test_threshold_and_patience_options():
    X = esn()
    default = information_processing_capacity(X, U, offset=W)
    strict = information_processing_capacity(X, U, offset=W, threshold=0.2)
    assert min(strict.capacities) > 0.2 and strict.total < default.total
    everything = information_processing_capacity(X, U, offset=W, use_threshold=False, max_degree=1, max_delay=20)
    assert len(everything.capacities) == 20


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a C++ compiler")
def test_python_matches_cpp(tmp_path):
    binary = os.path.join(ROOT, "MC")
    if not os.path.exists(binary):
        res = subprocess.run(["make", "-C", ROOT], capture_output=True, text=True)
        if res.returncode != 0:
            pytest.skip("C++ build failed (Armadillo / Boost missing?)")
    X = esn(bias=0.2)
    np.savetxt(tmp_path / "stateData", X)
    np.savetxt(tmp_path / "inputData", U)
    subprocess.run([binary, "-start_target_row", str(W), "-precision", "12"], cwd=tmp_path, check=True,
                   capture_output=True)
    ref = np.loadtxt(tmp_path / "memoryData", skiprows=1)
    r = information_processing_capacity(X, U, offset=W)
    write_table(r, str(tmp_path / "py.txt"), 3, precision=12)
    mine = np.loadtxt(tmp_path / "py.txt", skiprows=1)
    assert ref.shape == mine.shape
    np.testing.assert_array_equal(ref[:, :6], mine[:, :6])  # same significant targets, same order
    np.testing.assert_allclose(ref[:, 6:], mine[:, 6:], rtol=1e-7, atol=1e-9)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a C++ compiler")
def test_cpp_rejects_bad_input(tmp_path):
    binary = os.path.join(ROOT, "MC")
    if not os.path.exists(binary):
        pytest.skip("MC not built")
    np.savetxt(tmp_path / "stateData", np.stack([delayed(0)], 1))
    np.savetxt(tmp_path / "inputData", U[:500])
    res = subprocess.run([binary], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 1 and "input file has 500 values" in res.stderr
    res = subprocess.run([binary, "-state_file", "missing"], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 1 and "could not read the state matrix" in res.stderr
