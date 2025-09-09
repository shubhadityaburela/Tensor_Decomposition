import numpy as np
from numba import njit


# ============================================================================ #
#                       Type of basis functions and shifts                     #
# ============================================================================ #

@njit
def gaussian_exponent(x, mu, sigma=3.0):
    return (x - mu) / sigma ** 2.0


@njit
def gaussian(x, mu, sigma=3.0):
    return np.exp(-((x - mu) ** 2.0) / (2 * (sigma ** 2.0)))


def polynomial(c, t):
    return np.polyval(c, t)


def cubic(x, mu):
    return (x - mu) ** 3


def cubic_der(x, mu):
    return 3 * ((x - mu) ** 2)


@njit
def numba_polyval(coeffs, t_array):
    results = np.zeros_like(t_array)
    for i in range(t_array.shape[0]):
        t = t_array[i]
        result = 0.0
        for coeff in coeffs:
            result = result * t + coeff
        results[i] = result
    return results
