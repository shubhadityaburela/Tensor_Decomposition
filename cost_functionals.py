from minimizer_helper import generate_phi, Q_recons
from preliminaries import *


# ============================================================================ #
#                     FUNCTIONS FOR COST FUNCTIONAL                            #
# ============================================================================ #
def H(Q, alpha, beta, param):
    phiBeta = generate_phi(param, beta)
    return np.linalg.norm(Q - Q_recons(phiBeta, alpha), ord='fro') ** 2


def g(alpha, param):
    return param.alpha_solver_lamda_1 * np.linalg.norm(alpha, ord=1)


def f(beta, param):
    return param.beta_solver_lamda_2 * np.linalg.norm(beta, ord=1)


def f_tilde():
    return 0


def J(Q, alpha, beta, param):
    """
    Compute the value of the objective function J
    """
    return H(Q, alpha, beta, param) + g(alpha, param) + f(beta, param) + f_tilde()