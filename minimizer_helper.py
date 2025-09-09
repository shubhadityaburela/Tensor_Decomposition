import numpy as np

from preliminaries import *
import opt_einsum as oe


# ============================================================================ #
#                    Auxiliary Functions and Structures                        #
# ============================================================================ #
def shift(param, beta, nf):
    if param.type_shift[nf] == "polynomial":
        return numba_polyval(beta, param.t)
    elif param.type_shift[nf] == "sine":
        print("Sine shift not implemented yet")
        exit()
    elif param.type_shift[nf] == "polynomial+sine":
        print("Polynomial and sine shift mixture not implemented yet")
        exit()


def basis(param, shift, nf, k):
    if param.type_basis[nf, k] == "gaussian":
        X, S = np.meshgrid(param.x, shift)
        return gaussian(X.T, S.T, param.sigma[nf, k])
    elif param.type_basis[nf, k] == "cubic":
        X, S = np.meshgrid(param.x, shift)
        return cubic(X.T, S.T)
    else:
        print("Other basis not implemented yet")
        exit()


def facJac(param, shift, nf, k, d):
    if param.type_basis[nf, k] == "gaussian":
        X, S = np.meshgrid(param.x, shift)
        pre = oe.contract('ij,ij->ij', basis(param, shift, nf, k), gaussian_exponent(X.T, S.T, param.sigma[nf, k]))
        if param.type_shift[nf] == "polynomial":
            return oe.contract('ij,j->ij', pre, param.t ** (param.degree[nf] - d - 1))
        elif param.type_shift[nf] == "sine":
            print("Sine shift not implemented yet")
            exit()
        elif param.type_shift[nf] == "polynomial+sine":
            print("Polynomial and sine shift mixture not implemented yet")
            exit()
    elif param.type_basis[nf, k] == "cubic":
        X, S = np.meshgrid(param.x, shift)
        pre = cubic_der(X.T, S.T)
        if param.type_shift[nf] == "polynomial":
            return oe.contract('ij,j->ij', pre, param.t ** (param.degree[nf] - d - 1))
        elif param.type_shift[nf] == "sine":
            print("Sine shift not implemented yet")
            exit()
        elif param.type_shift[nf] == "polynomial+sine":
            print("Polynomial and sine shift mixture not implemented yet")
            exit()
    else:
        print("Other basis not implemented yet")
        exit()



def generate_phi(param, beta):
    phiBeta = np.zeros((sum(param.K), param.n, param.m))
    idx = 0
    for nf in range(param.nf):
        beta_f = beta[param.degree_st[nf]:param.degree_st[nf + 1]]
        shift_f = param.center_of_matrix[nf] + shift(param, beta_f, nf)
        for k in range(param.K[nf]):
            phiBeta[idx, ...] = basis(param, shift_f, nf, k)
            idx = idx + 1

    return phiBeta



def generate_phiJac(param, beta):
    phiBeta = np.zeros((sum(param.K), param.n, param.m))
    phiBetaJac = []
    idx1 = 0
    for nf in range(param.nf):
        phiBetaJac_f = np.zeros((param.n, param.m, param.K[nf], param.degree[nf]))
        beta_f = beta[param.degree_st[nf]:param.degree_st[nf + 1]]
        shift_f = param.center_of_matrix[nf] + shift(param, beta_f, nf)
        for k in range(param.K[nf]):
            phiBeta[idx1, ...] = basis(param, shift_f, nf, k)
            idx1 = idx1 + 1
            for d in range(param.degree[nf]):
                phiBetaJac_f[..., k, d] = facJac(param, shift_f, nf, k, d)

        phiBetaJac.append(phiBetaJac_f)

    return phiBeta, phiBetaJac



@njit
def prox_l1(data, reg_param):
    tmp = np.abs(data) - reg_param
    tmp = (tmp + np.abs(tmp)) / 2
    y = np.sign(data) * tmp
    return y


def Q_recons(phiBeta, alpha):

    Q = oe.contract('ijk,ik->jk', phiBeta, alpha)

    return Q



def reconstruction_error(Q, alpha, beta, param):
    phiBeta = generate_phi(param, beta)
    return np.linalg.norm(Q - Q_recons(phiBeta, alpha)) / np.linalg.norm(Q)
