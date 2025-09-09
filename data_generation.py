from preliminaries import *


# ============================================================================ #
#                          Data generation functions                           #
# ============================================================================ #
def generate_data_single_wave(param):
    shift = numba_polyval(param.beta, param.t)
    mu = np.asarray(param.center_of_matrix + shift, dtype=np.float64)
    X, MU = np.meshgrid(param.x, mu)
    Q = gaussian(X.T, MU.T, param.sigma)
    return Q


def generate_data_crossing_wave(param):
    shift1 = numba_polyval(param.beta[0], param.t)
    shift2 = numba_polyval(param.beta[1], param.t)
    mu1 = np.asarray(param.center_of_matrix[0] + shift1, dtype=np.float64)
    mu2 = np.asarray(param.center_of_matrix[1] + shift2, dtype=np.float64)
    X1, MU1 = np.meshgrid(param.x, mu1)
    X2, MU2 = np.meshgrid(param.x, mu2)

    Q1 = gaussian(X1.T, MU1.T, param.sigma[0, 0])
    Q2 = gaussian(X2.T, MU2.T, param.sigma[1, 0])

    Q = Q1 + Q2

    return Q, Q1, Q2



def generate_data_crossing_wave_sine(param):
    t_max = param.t[-1]
    shift1 = numba_polyval(param.beta[0], param.t)
    shift2 = numba_polyval(param.beta[1], param.t)
    phase1 = np.sin(np.pi * param.t / t_max)[None, :]
    phase2 = np.cos(np.pi * param.t / t_max)[None, :]
    mu1 = np.asarray(param.center_of_matrix[0] + shift1, dtype=np.float64)
    mu2 = np.asarray(param.center_of_matrix[1] + shift2, dtype=np.float64)
    X1, MU1 = np.meshgrid(param.x, mu1)
    X2, MU2 = np.meshgrid(param.x, mu2)

    Q1 = phase1 * gaussian(X1.T, MU1.T, param.sigma[0, 0])
    Q2 = phase2 * gaussian(X2.T, MU2.T, param.sigma[1, 0])

    Q = Q1 + Q2

    return Q, Q1, Q2


def generate_data_faded(param, beta):
    param.x = np.arange(0, param.n)
    param.t = np.linspace(param.t_start, param.t_end, param.m)

    # Construct the fading of the waves
    damp = np.arange(len(param.t))
    damp1 = damp[::-1] / np.max(damp) + 0.3
    damp2 = (damp / (np.max(damp) / 2)) + 2.5

    q1 = np.zeros((param.n, param.m))
    q2 = np.zeros((param.n, param.m))
    shift1 = np.polyval(beta[0], param.t)
    shift2 = np.polyval(beta[1], param.t)
    for col in range(param.m):
        sigma_t = 1.5
        q1[:, col] = gaussian(param.x, param.center_of_matrix[0] + shift1[col], sigma_t * damp1[col])
        q2[:, col] = gaussian(param.x, param.center_of_matrix[1] + shift2[col], sigma_t * damp2[col])

    Q = np.maximum(q1, q2)

    return Q, q1, q2


def generate_data_sine(param, beta1, beta2):
    param.x = np.arange(0, param.n)
    param.t = np.linspace(param.t_start, param.t_end, param.m)

    shift1 = beta1[0] * np.sin(beta1[1] * np.pi * param.t)
    shift2 = np.polyval(beta2, param.t)

    q1 = np.zeros((param.n, param.m))
    q2 = np.zeros((param.n, param.m))

    for col in range(param.m):
        q1[:, col] = gaussian(param.x, param.center_of_matrix1 + shift1[col])
        q2[:, col] = gaussian(param.x, param.center_of_matrix2 + shift2[col])

    Q = np.maximum(q1, q2)

    return Q, q1, q2

