
import numpy as np
import numpy.linalg as la
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
from scipy.linalg import cholesky


def rls(x, A, y):
    # initialization
    xhat = np.zeros_like(x)
    mr, nt = A.shape
    Gamma = np.eye(nt)
    mse = np.zeros(mr)
    for n in range(mr):
        hn = A[n, :].reshape(-1, 1)
        z = Gamma @ np.conj(hn)
        alpha = 1 / (1 + hn.T @ z)
        Gamma = Gamma - alpha * z @ np.conj(z).T
        e = y[n] - hn.T @ xhat
        xhat = xhat + alpha * e * z
        mse[n] = np.mean(abs(x - xhat) ** 2)

    return xhat, mse


def sgd(x, A, y):
    # initialization
    xhat = np.zeros_like(x)
    mr, nt = A.shape
    mse = np.zeros(mr)
    step_size = 2

    for n in range(mr):
        hn = A[n, :].reshape(-1, 1)
        e = y[n] - hn.T @ xhat
        xhat = xhat + step_size * np.conj(hn) * e
        mse[n] = np.mean(abs(x - xhat) ** 2)

    return xhat, mse


def asgd(x, A, y):
    # initialization
    xhat = np.zeros_like(x)
    mr, nt = A.shape
    mse = np.zeros(mr)
    s = xhat.copy()
    n0 = 100

    for n in range(mr):
        hn = A[n, :].reshape(-1, 1)
        e = y[n] - hn.T @ xhat
        step_size = 1 / np.real(np.conj(hn.T) @ hn)
        s = s + step_size * np.conj(hn) * e
        if n < n0:
            xhat = s.copy()
        else:
            xhat = xhat + 1 / (n - n0 + 1) * (s - xhat)
        mse[n] = np.mean(abs(x - xhat) ** 2)

    return xhat, mse
