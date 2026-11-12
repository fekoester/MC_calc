"""
Information processing capacity (nonlinear memory capacity) of a reservoir, in NumPy.

Same algorithm and defaults as the C++ program in this repository (MC.cpp): for every target
y_t = prod_j P_{d_j}(u_{t - k_j}) (products of Legendre polynomials of delayed inputs), the capacity is

    C[y] = y^T X (X^T X + lambda I)^+ X^T y / ||y||^2      (X and y centred)

Capacities are summed per total degree d_1 + d_2 + ... A capacity counts only if it exceeds a significance
threshold (factor * chi2_quantile(N, 1 - p) / T by default), and a delay scan stops after `patience`
consecutive sub-threshold delays.
"""

from .capacity import CapacityResult, information_processing_capacity, legendre_table, write_table

__all__ = ["information_processing_capacity", "CapacityResult", "legendre_table", "write_table"]
__version__ = "0.2.0"
