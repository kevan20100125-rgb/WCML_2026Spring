
import numpy as np
import numpy.linalg as la
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
from scipy.linalg import cholesky


def mhcg(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AH = np.conj(A).T
    AHA = AH @ A
    grad_preconditioner = la.inv(AHA + noise_var / (dqam ** 2) * np.eye(nt))  # note: dqam^2, otherwise gamma is small
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam
    x_sample = np.zeros((nt, samplers), dtype=complex)  # from the parallel samplers
    r_norm_sample = np.zeros(samplers)
    mse = np.zeros(samplers)
    if lr_approx is False:
        p_mat = A @ grad_preconditioner @ AH
    else:
        p_mat = None

    if mmse_init:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        x_mmse = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
    else:
        x_mmse = None
    # parallel samplers
    for s in range(samplers):
        # z0 = np.random.uniform(low=-1, high=1, size=(nt, 1)) + 1j * np.random.uniform(low=-1, high=1, size=(nt, 1))
        if mmse_init:
            xhat = x_mmse.copy()
        else:
            xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(nt, 1))].copy()
        # x_survivor = xhat.copy()
        r = y - A @ xhat
        r_norm = la.norm(r) ** 2
        if lr_approx is False:
            pr_prev = p_mat @ r
            lr = np.real(np.conj(r.T) @ pr_prev) / la.norm(pr_prev) ** 2
        else:
            lr = 1

        # core
        for t in range(iter):
            # construct the proposal
            xhat = xhat + lr * grad_preconditioner @ AH @ r
            # x_prop = constellation_norm[np.argmin(abs(z_prop * np.ones((nt, 2 ** mu)) - constellation_norm),
            #                                       axis=1)].reshape(-1, 1)  # quantization
            r = y - A @ xhat
            r_norm = la.norm(r) ** 2
            # if r_norm_prop < r_norm:
            #     x_survivor = x_prop.copy()

        x_sample[:, s] = xhat.reshape(-1)
        r_norm_sample[s] = r_norm
        mse[s] = np.mean(abs(xhat - x) ** 2)
        # x_sample[:, s] = x_survivor.reshape(-1)
        # r_norm_sample[s] = la.norm(y - A @ x_survivor) ** 2
        # mse[s] = np.mean(abs(x_survivor - x) ** 2)

    # select the sample that minimizes the ML cost
    xhat = x_sample[:, np.argmin(r_norm_sample)].reshape(-1, 1)

    return xhat, mse


def mhcg_para(x, A, y, noise_var, mu=2, iter=8, samplers=16, mmse_init=False, vec_step_size=True):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    noise_scale = (2 * dqam) ** 2 / np.log(2)  # change to the neighbor be given a acceptance rate of 50%
    AH = np.conj(A).T
    AHA = AH @ A
    ytilde = AH @ y  # (nt, 1)
    xi = AHA + noise_var * np.eye(nt)
    ones = np.ones((samplers, nt, 2 ** mu))
    if mr != nt:
        Ainv = cholesky(la.inv(AHA))
    else:
        Ainv = la.inv(A)
    col_norm = 1 / la.norm(Ainv, axis=0)
    covar = Ainv * col_norm
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam

    if mmse_init:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()

    xhat_cg = xhat.copy()  # continuous value
    r_cg = ytilde - xi @ xhat_cg  # (np, nt, 1)
    r_norm_cg = np.real(np.conj(np.transpose(r_cg, axes=[0, 2, 1])) @ r_cg)  # (np, 1, 1)
    di = r_cg.copy()

    # core
    for t in range(iter):
        # compute the approximate solution based on prior conjugate direction and residual
        xi_di = xi @ di
        alpha = r_norm_cg / np.real((np.conj(np.transpose(di, axes=[0, 2, 1])) @ xi_di))  # (np, 1, 1)
        xhat_cg = xhat_cg + alpha * di  # (np, nt, 1)

        if t % nt >= (nt // 2):
            xtmp = constellation_norm[np.argmin(abs(xhat_cg * ones - constellation_norm),
                                                axis=2)].reshape(-1, nt, 1)  # quantization
            rtmp = y - A @ xtmp
            r_norm_tmp = np.real(np.conj(np.transpose(rtmp, axes=[0, 2, 1])) @ rtmp)
            if vec_step_size:
                step_size = np.maximum(dqam, abs(rtmp) / np.sqrt(nt))  # (np, nt, 1)
            else:
                step_size = np.maximum(dqam, np.sqrt(r_norm_tmp / nt))  # (np, 1, 1)
            for i in range(2):
                # construct the proposal
                v = (np.random.randn(samplers, nt, 1) + 1j * np.random.randn(samplers, nt, 1)) / np.sqrt(
                    2)  # zero-mean, unit-variance
                xtmp = xhat_cg + step_size * covar @ v
                xtmp = constellation_norm[np.argmin(abs(xtmp * ones - constellation_norm),
                                                    axis=2)].reshape(-1, nt, 1)  # quantization
                # update survivor
                rtmp = y - A @ xtmp
                r_norm_tmp = np.real(np.conj(np.transpose(rtmp, axes=[0, 2, 1])) @ rtmp)
                update = np.squeeze(r_norm_survivor > r_norm_tmp)
                if update.any():
                    x_survivor[update] = xtmp[update]
                    r_norm_survivor[update] = r_norm_tmp[update]
                # acceptance step
                log_pacc = np.minimum(0, - (r_norm_tmp - r_norm) / (2 * noise_var))  # (np, 1, 1)
                p_acc = np.exp(log_pacc)
                p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
                index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
                if index.any():
                    xhat[index], r[index], r_norm[index] = xtmp[index], rtmp[index], r_norm_tmp[index]
                if vec_step_size:
                    step_size = np.maximum(dqam, abs(r) / np.sqrt(nt))  # (np, nt, 1)
                else:
                    step_size = np.maximum(dqam, np.sqrt(r_norm / nt))  # (np, 1, 1)

        # compute conjugate direction and residual
        r_cg -= alpha * xi_di
        r_norm_cg_last = r_norm_cg.copy()
        r_norm_cg = np.real(np.conj(np.transpose(r_cg, axes=[0, 2, 1])) @ r_cg)
        beta = r_norm_cg / r_norm_cg_last  # (np, 1, 1)
        di = r_cg + beta * di  # (np, nt, 1)

        if ((t + 1) % nt) == 0:  # restart CG
            xhat_cg = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
            r_cg = ytilde - xi @ xhat_cg
            r_norm_cg = np.real(np.conj(np.transpose(r_cg, axes=[0, 2, 1])) @ r_cg)  # (np, 1, 1)
            di = r_cg.copy()

    # mse = np.squeeze(np.mean(abs(xhat - x) ** 2, axis=1))
    # # select the sample that minimizes the ML cost
    # x_hat = xhat[np.argmin(r_norm), :, :].copy()
    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=1))
    # r = y - A @ x_survivor
    # r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse


def incomplete_cholesky(mat, ratio=None):
    n = mat.shape[1]
    eps = 1 / 16
    L = np.zeros((n, n), dtype=complex)
    for j in range(n):
        # mat[j, j] = np.sqrt(mat[j, j])
        # for i in range(j+1, n):
        #     if abs(mat[i, j]) > eps * abs(mat[j, j]):
        #         mat[i, j] = mat[i, j] / mat[j, j]
        for i in range(j, n):
            if abs(mat[i, j]) <= eps * abs(mat[j, j]):
                L[i, j] = 0
            else:
                sum = 0
                for k in range(j):
                    sum += L[i, k] * np.conj(L[j, k])
                if i == j:
                    L[i, j] = np.sqrt(mat[i, j] - sum)
                else:
                    L[i, j] = (mat[i, j] - sum) / L[j, j]
    return L


def icholesky(a):
    n = a.shape[1]
    eps = 1 / 32
    for j in range(n):
        a[j, j] = np.sqrt(a[j, j])
        for i in range(j+1, n):
            if a[i, j] != 0:
                a[i, j] = a[i, j] / a[j, j]

        for k in range(j+1, n):
            for i in range(k, n):
                if a[i, k] != 0:
                    a[i, k] = a[i, k] - a[i, j] * np.conj(a[k, j])

    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = 0

    return a

