# This file is modified from https://github.com/VIP-Group/DBP
import numpy as np
import numpy.linalg as la
from .utils import NLE


def dcg(x, A, y, c, noise_var, iter=8, det='MMSE'):
    # initialization
    mr, nt = A.shape
    n = mr // c
    yc = y.reshape(c, n, 1)
    Ac = A.reshape(c, n, nt)
    AcH = np.conj(np.transpose(Ac, axes=[0, 2, 1]))  # (c, nt, n)

    # distributed preprocessing (iteration 1)
    y_mrc = AcH @ yc  # (c, nt, 1)  compute local MRC

    # centralized processing
    r = np.sum(y_mrc, axis=0)  # (nt, 1)
    di = r.copy()
    r_norm2 = np.conj(r.T) @ r
    xhat = np.zeros_like(x, dtype=complex)

    # inner loop (iteration 2, 3, ...)
    for i in range(iter):
        # decentralized matrix processing
        wc = AcH @ (Ac @ di)  # (c, nt, 1)

        # centralized processing
        if det == 'MMSE':
            xi_di = np.sum(wc, axis=0) + noise_var * di
        elif det == 'ZF':
            xi_di = np.sum(wc, axis=0)
        else:
            raise RuntimeError('The selected detector does not exist!')

        # conventional CG update
        alpha = r_norm2 / (np.conj(di).T @ xi_di)
        xhat += alpha * di
        r -= alpha * xi_di
        r_norm2_last = r_norm2.copy()
        r_norm2 = np.conj(r.T) @ r
        beta = r_norm2 / r_norm2_last
        di = r + beta * di

    return xhat


def dep_real(x, A, y, c, noise_var, T=10, mu=2):
    # initialization
    beta = 0.2
    mr, nt = A.shape
    n = mr // c
    yc = y.reshape(c, n, 1)
    Ac = A.reshape(c, n, nt)
    AcT = np.transpose(Ac, axes=[0, 2, 1])  # (c, nt, n)
    AcTAc = AcT @ Ac  # (c, nt, nt)
    AcTyc = AcT @ yc  # (c, nt, 1)
    Lambda = 2 * np.ones((c, nt, 1))
    gamma = np.zeros((c, nt, 1))
    MSE = np.zeros(T)
    diag = np.zeros((c, nt, 1))
    ub = 0

    for t in range(T):
        # compute the approximated posterior mean and covariance by LMMSE (module A in APs)
        Sigma = la.inv(AcTAc + noise_var * Lambda * np.eye(nt))  # (c, nt, nt)
        Mu = Sigma @ (AcTyc + noise_var * gamma)  # (c, nt, 1)

        # compute the extrinsic mean and covariance (module B in CPU)
        for k in range(c):
            diag[k, :, :] = noise_var * np.diag(Sigma[k]).reshape(nt, 1)
        vab_c = diag / (1 - diag * Lambda)  # (c, nt, 1)
        vab_c = np.maximum(vab_c, 5e-7)
        uab_c = vab_c * (Mu / diag - gamma)  # (c, nt, 1)

        # MRC
        vab = 1 / np.sum(1 / vab_c, axis=0)  # (nt, 1)
        uab = vab * np.sum(uab_c / vab_c, axis=0)  # (nt, 1)

        # compute the posterior mean and variance
        _, _, ub, vb = NLE(vab, uab, orth=False, mu=mu, EP=True, norm=np.sqrt(1))  # (nt, 1)
        vb = np.maximum(vb, 5e-13)
        MSE[t] = np.mean((x - ub) ** 2)

        # update the extrinsic mean and variance
        gamma_last = gamma
        Lambda_last = Lambda
        Lambda = (vab_c - vb) / vab_c / vb  # (c, nt, 1)
        gamma = (ub * vab_c - uab_c * vb) / vab_c / vb  # (c, nt, 1)
        idx = Lambda < 0  # (c, nt, 1)
        Lambda[idx] = Lambda_last[idx]
        gamma[idx] = gamma_last[idx]
        # damping
        Lambda = beta * Lambda + (1 - beta) * Lambda_last
        gamma = beta * gamma + (1 - beta) * gamma_last

    return ub, MSE

def dnewton(x, A, y, c, iter=4, hessian_approx=True):
    # initialization
    mr, nt = A.shape
    n = mr // c
    yc = y.reshape(c, n, 1)
    Ac = A.reshape(c, n, nt)
    AcH = np.conj(np.transpose(Ac, axes=[0, 2, 1]))  # (c, nt, n)
    AcHAc = AcH @ Ac  # (c, nt, nt)

    mse = np.zeros(iter)
    # initial iteration
    if hessian_approx:  # only effective when n >> nt ?
        dc = np.transpose(la.norm(Ac, axis=1, keepdims=True) ** 2, axes=[0, 2, 1])  # (c, nt, 1)
        d = np.squeeze(np.sum(dc, axis=0))  # (nt, )
        hessian_inv = np.diag(1 / d)  # (nt, nt)
        x_c = 1 / dc * (AcH @ yc)  # (c, nt, 1)
    else:
        hessian_inv = la.inv(np.sum(AcHAc, axis=0))  # (nt, nt)
        x_c = la.inv(AcHAc) @ (AcH @ yc)  # (c, nt, 1)
    xhat = x_c[c - 1] + hessian_inv @ np.sum(AcH @ (yc - Ac @ x_c), axis=0)  # (nt, 1)
    # xhat = np.zeros_like(x)
    mse[0] = np.mean(abs(x - xhat) ** 2)

    # core (iteration 2, 3...)
    for t in range(iter - 1):
        grad = np.sum(AcH @ (yc - Ac @ xhat), axis=0)
        xhat = xhat + hessian_inv @ grad
        mse[t + 1] = np.mean(abs(x - xhat) ** 2)

    return xhat, mse
