
import numpy as np
import numpy.linalg as la
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
from scipy.linalg import cholesky, sqrtm
import time

qam_order = 16
gamma = - np.log(np.sqrt(2 * (qam_order - 1) / 3) - 1) * np.ones(20)
LR = np.ones(20)
para = {}

# loading the trained model parameters
try:
    for k,v in np.load("model/MHNGD_8x8_16QAM_25dB"
                       "/MHNGD_8x8_16QAM_25dB_final_norm_ns12_np8_nolr_dtune_sigmoid_sur_var1_para_gd8_.npz").items():
        para[k] = v
except IOError:
    print("no such file")
    pass
# get parameters for MHGD Network
for t in range(20):
    if para.get("gamma_"+str(t)+":0",-1) != -1:
        gamma[t] = para["gamma_"+str(t)+":0"]
    if para.get("lr_"+str(t)+":0",-1) != -1:
        LR[t] = para["lr_"+str(t)+":0"]


def mhgd(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AH = np.conj(A).T
    AHA = AH @ A
    grad_preconditioner = la.inv(AHA + noise_var / (dqam ** 2) * np.eye(nt))  # note: dqam^2, otherwise gamma is small
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
        step_size = max(dqam, np.sqrt(r_norm / nt))

        # core
        for t in range(iter):
            v = (np.random.randn(nt, 1) + 1j * np.random.randn(nt, 1)) / np.sqrt(2)  # zero-mean, unit-variance
            # construct the proposal
            z_prop = xhat + lr * grad_preconditioner @ AH @ r + step_size * covar @ v
            x_prop = constellation_norm[np.argmin(abs(z_prop * np.ones((nt, 2 ** mu)) - constellation_norm),
                                                  axis=1)].reshape(-1, 1)  # quantization
            r_prop = y - A @ x_prop
            r_norm_prop = la.norm(r_prop) ** 2
            # if r_norm_prop < r_norm:
            #     x_survivor = x_prop.copy()

            # acceptance step
            log_pacc = min(0, - (r_norm_prop - r_norm) / (2 * noise_var))
            p_acc = np.exp(log_pacc)
            p_uni = np.random.uniform(low=0.0, high=1.0)
            if p_acc >= p_uni:
                xhat, r, r_norm = x_prop.copy(), r_prop.copy(), r_norm_prop.copy()
                if lr_approx is False:
                    pr_prev = p_mat @ r
                    lr = np.real(np.conj(r.T) @ pr_prev) / la.norm(pr_prev) ** 2
                step_size = max(dqam, np.sqrt(r_norm / nt))

        x_sample[:, s] = xhat.reshape(-1)
        r_norm_sample[s] = r_norm
        mse[s] = np.mean(abs(xhat - x) ** 2)
        # x_sample[:, s] = x_survivor.reshape(-1)
        # r_norm_sample[s] = la.norm(y - A @ x_survivor) ** 2
        # mse[s] = np.mean(abs(x_survivor - x) ** 2)

    # select the sample that minimizes the ML cost
    xhat = x_sample[:, np.argmin(r_norm_sample)].reshape(-1, 1)

    return xhat, mse


def cg(x_init, xi, y, iter=8):
    r = y - xi @ x_init
    di = r.copy()
    xhat = x_init
    r_norm = la.norm(r)
    for i in range(iter):
        # compute the approximate solution based on prior conjugate direction and residual
        xi_di = xi @ di
        alpha = r_norm ** 2 / (np.conj(di).T @ xi_di)
        xhat += alpha * di

        # compute conjugate direction and residual
        r -= alpha * xi_di
        r_norm_last = r_norm.copy()
        r_norm = la.norm(r)
        beta = r_norm ** 2 / r_norm_last ** 2
        di = r + beta * di
        if r_norm < 1e-4:
            break

    return xhat


def mhgd_para(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False, vec_step_size=True,
              constellation_norm=None, hessian_approx=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    noise_scale = (2 * dqam) ** 2 / np.log(2)  # change to the neighbor be given a acceptance rate of 50%
    AH = np.conj(A).T
    if hessian_approx:
        d = np.sum(np.conj(A) * A, axis=0, keepdims=True)  # (1, nt)
        AHA = d * np.eye(nt)
    else:
        AHA = AH @ A
    grad_preconditioner = la.inv(AHA + noise_var / (dqam ** 2) * np.eye(nt))  # note: dqam^2, otherwise gamma is small
    # alpha = 0.5 / ((nt / 64) ** (1 / 2))
    alpha = 1 / ((nt / 8) ** (1 / 3))
    # alpha = 0.5 / (((mr * nt) / (64 * 8)) ** (1 / 2))
    # alpha = 0.79
    ones = np.ones((samplers, nt, 2 ** mu))
    if mr != nt:
        Ainv = (cholesky(la.inv(AHA), lower=True))  # choice 1: lower triangular matrix
        # Ainv = la.inv(AHA)  # choice 2: inv(AHA)
        # Ainv = sqrtm(Ainv)  # choice 4: sqrtm(inv(AHA))
        col_norm = 1 / la.norm(Ainv, axis=1, keepdims=True)
        covar = Ainv * col_norm
        # covar = np.sqrt(covar)  # choice 3: sqrt(norm(inv(AHA)))
        # covar = np.eye(nt)  # choice 5: none
    else:
        Ainv = la.inv(A)
        col_norm = 1 / la.norm(Ainv, axis=0)
        covar = Ainv * col_norm
    # covar = np.eye(nt)  # choice 5: none
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    elif mu == 6:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = constellation_norm * dqam

    if lr_approx is False:
        p_mat = A @ grad_preconditioner @ AH  # (nr, nr)
    else:
        p_mat = None

    if mmse_init is True:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        # xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
        # xhat[np.random.randint(low=0, high=samplers)] = xmmse  # only one sampler initialized by MMSE
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    elif mmse_init == 'cg':
        xinit = np.zeros_like(x)
        x_mmse = cg(xinit, xi=AHA + noise_var * np.eye(nt), y=AH @ y, iter=nt)
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                            axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()
    if lr_approx is False:
        pr_prev = p_mat @ r  # (np, nr, 1)
        lr = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ pr_prev /
             (np.conj(np.transpose(pr_prev, axes=[0, 2, 1])) @ pr_prev))  # (np, 1, 1)
    else:
        lr = 1
    if vec_step_size:
        step_size = np.maximum(dqam, abs(r) / np.sqrt(mr))  # (np, nt, 1)
    else:
        step_size = np.maximum(dqam, np.sqrt(r_norm / mr)) * alpha  # (np, 1, 1)

    # core
    acc_rate = 0
    for t in range(iter):
        # sta = time.time()
        v = (np.random.randn(samplers, nt, 1) + 1j * np.random.randn(samplers, nt, 1)) / np.sqrt(2)  # zero-mean, unit-variance
        # construct the proposal
        z_grad = xhat + lr * (grad_preconditioner @ (AH @ r))  # (np, nt, 1)
        # t1 = time.time() - sta
        # if vec_step_size:
        #     step_size = np.maximum(dqam, abs(y - A @ z_grad) / np.sqrt(nt))  # (np, nt, 1)
        # else:
        #     step_size = np.maximum(dqam, la.norm(y - A @ z_grad, axis=1, keepdims=True) / np.sqrt(nt))  # (np, 1, 1)
        z_prop = z_grad + step_size * (covar @ v) # (np, nt, 1)
        # t2 = time.time() - sta - t1
        x_prop = constellation_norm[np.argmin(abs(z_prop * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
        # t3 = time.time() - sta - t1 - t2
        # x_prop = xhat + lr * grad_preconditioner @ AH @ r
        r_prop = y - A @ x_prop
        r_norm_prop = np.real(np.conj(np.transpose(r_prop, axes=[0, 2, 1])) @ r_prop)  # (np, 1, 1)
        # t4 = time.time() - sta - t1 - t2 - t3
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]
        # t5 = time.time() - sta - t1 - t2 - t3 - t4

        # acceptance step todo: noise scaling
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (1)) # (np, 1, 1) noise_var + --> acceptance rate +
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        acc_rate += sum(index)
        if index.any():
            xhat[index], r[index], r_norm[index] = x_prop[index], r_prop[index], r_norm_prop[index]
        # t6 = time.time() - sta - t1 - t2 - t3 - t4 - t5
        if lr_approx is False and index.any():
            pr_prev = p_mat @ r[index]  # (np_update, nr, 1)
            lr[index] = np.real(np.conj(np.transpose(r[index], axes=[0, 2, 1])) @ pr_prev /
                        (np.conj(np.transpose(pr_prev, axes=[0, 2, 1])) @ pr_prev))  # (np_update, 1, 1)
        # t7 = time.time() - sta - t1 - t2 - t3 - t4 - t5 - t6
        if vec_step_size:
            step_size[index] = np.maximum(dqam, abs(r[index]) / np.sqrt(mr))  # (np_update, nt, 1)
        else:
            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index] / mr)) * alpha  # (np_update, 1, 1)
        # t8 = time.time() - sta - t1 - t2 - t3 - t4 - t5 - t6 - t7

    # mse = np.squeeze(np.mean(abs(xhat - x) ** 2, axis=1))
    # # select the sample that minimizes the ML cost
    # x_hat = xhat[np.argmin(r_norm), :, :].copy()
    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=1))
    # r = y - A @ x_survivor
    # r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse, (acc_rate / samplers / iter)


def mhgd_real(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False, vec_step_size=True,
              constellation_norm=None, hessian_approx=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AH = (A).T
    if hessian_approx:
        d = np.sum((A) * A, axis=0, keepdims=True)  # (1, nt)
        AHA = d * np.eye(nt)
    else:
        AHA = AH @ A
    grad_preconditioner = la.inv(AHA + noise_var / (dqam ** 2) * np.eye(nt))  # from the equivalent real-valued system
    # L = np.abs(A) * np.sum(np.abs(A), axis=1, keepdims=True)  # (nr, nt)
    # nu = np.amax(la.svd(A, compute_uv=False) ** 2)
    # Q = np.diag(1 / (noise_var * nu * (L.T @ np.ones((mr, 1))).reshape(-1)))
    # alpha = 0.5 / ((nt / 64) ** (1 / 2))
    alpha = 1 / ((nt / 8) ** (1 / 3))
    # alpha = 0.5 / (((mr * nt) / (64 * 8)) ** (1 / 2))
    # alpha = 1
    ones = np.ones((1, 1, 2 ** (mu // 2)))
    # if mr != nt:
    #     Ainv = (cholesky(la.inv(AHA), lower=True))  # choice 1: lower triangular matrix
    #     col_norm = 1 / la.norm(Ainv, axis=1, keepdims=True)
    #     covar = Ainv * col_norm
    # else:
    #     Ainv = la.inv(A)
    #     col_norm = 1 / la.norm(Ainv, axis=0)
    #     covar = Ainv * col_norm
    covar = np.eye(nt)  # choice 5: none
    if mu == 2:
        constellation_norm = np.array([-1, +1]) * dqam
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +1, +3]) * dqam
    elif mu == 6:
        constellation_norm = np.array([-7, -5, -3, -1, +1, +3, +5, +7]) * dqam
    else:
        constellation_norm = constellation_norm * dqam

    if lr_approx is False:
        p_mat = A @ grad_preconditioner @ AH  # (nr, nr)
    else:
        p_mat = None

    if mmse_init is True:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((1, 2 ** (mu // 2))) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        # xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
        # xhat[np.random.randint(low=0, high=samplers)] = xmmse  # only one sampler initialized by MMSE
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** (mu // 2), size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = ((np.transpose(r, axes=[0, 2, 1])) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()
    if lr_approx is False:
        pr_prev = p_mat @ r  # (np, nr, 1)
        lr = (np.transpose(r, axes=[0, 2, 1]) @ pr_prev /
             ((np.transpose(pr_prev, axes=[0, 2, 1])) @ pr_prev))  # (np, 1, 1)
    else:
        lr = 1
    if vec_step_size:
        step_size = np.maximum(dqam, abs(r) / np.sqrt(mr))  # (np, nt, 1)
    else:
        step_size = np.maximum(dqam, np.sqrt(r_norm / mr)) * alpha  # (np, 1, 1)
    # step_size = np.sqrt(2 * lr)
    # core
    acc_rate = 0
    for t in range(iter):
        v = (np.random.randn(samplers, nt, 1))# zero-mean, unit-variance
        # construct the proposal
        z_grad = xhat + lr * (grad_preconditioner @ (AH @ r))  # (np, nt, 1)
        z_prop = z_grad + step_size * (covar @ v) # (np, nt, 1)
        x_prop = constellation_norm[np.argmin(abs(z_prop * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
        r_prop = y - A @ x_prop
        r_norm_prop = ((np.transpose(r_prop, axes=[0, 2, 1])) @ r_prop)  # (np, 1, 1)
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]

        # acceptance step
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (2 * noise_var)) # (np, 1, 1) noise_var + --> acceptance rate +
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        acc_rate += sum(index)
        if index.any():
            xhat[index], r[index], r_norm[index] = x_prop[index], r_prop[index], r_norm_prop[index]
        if lr_approx is False and index.any():
            pr_prev = p_mat @ r[index]  # (np_update, nr, 1)
            lr[index] = ((np.transpose(r[index], axes=[0, 2, 1])) @ pr_prev /
                        ((np.transpose(pr_prev, axes=[0, 2, 1])) @ pr_prev))  # (np_update, 1, 1)
        if vec_step_size:
            step_size[index] = np.maximum(dqam, abs(r[index]) / np.sqrt(mr))  # (np_update, nt, 1)
        else:
            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index] / mr)) * alpha  # (np_update, 1, 1)
        # step_size = np.sqrt(2 * lr)
    # select the sample that minimizes the ML cost
    # x_hat = xhat[np.argmin(r_norm), :, :].copy()
    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=1))
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse, (acc_rate / samplers / iter)


def mhgd_para_temp(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False, vec_step_size=True):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    noise_scale = (2 * dqam) ** 2 / np.log(2)  # change to the neighbor be given a acceptance rate of 50%
    AH = np.conj(A).T
    AHA = AH @ A
    grad_preconditioner = la.inv(AHA + noise_var / (dqam ** 2) * np.eye(nt))  # note: dqam^2, otherwise gamma is small
    alpha = 1
    ones = np.ones((samplers, nt, 2 ** mu))
    if mr != nt:
        # Ainv = (cholesky(la.inv(AHA), lower=True)) ** (1/4)  # choice 1: lower triangular matrix
        # Ainv = la.inv(AHA)  # choice 2: inv(AHA)
        # Ainv = sqrtm(Ainv)  # choice 4: sqrtm(inv(AHA))
        # col_norm = 1 / la.norm(Ainv, axis=0)
        # covar = Ainv * col_norm
        # covar = np.sqrt(covar)  # choice 3: sqrt(norm(inv(AHA)))
        covar = np.eye(nt)  # choice 5: none
    else:
        Ainv = la.inv(A)
        col_norm = 1 / la.norm(Ainv, axis=0)
        covar = Ainv * col_norm
        # covar = np.eye(nt)  # choice 5: none
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam

    if lr_approx is False:
        p_mat = A @ grad_preconditioner @ AH  # (nr, nr)
    else:
        p_mat = None

    if mmse_init is True:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    elif mmse_init == 'cg':
        xinit = np.zeros_like(x)
        x_mmse = cg(xinit, xi=AHA + noise_var * np.eye(nt), y=AH @ y, iter=nt)
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                            axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()
    if lr_approx is False:
        pr_prev = p_mat @ r  # (np, nr, 1)
        lr = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ pr_prev /
             (np.conj(np.transpose(pr_prev, axes=[0, 2, 1])) @ pr_prev))  # (np, 1, 1)
    else:
        lr = 1
    if vec_step_size:
        step_size = np.maximum(dqam, abs(AH @ r) / np.sqrt(nt))  # (np, nt, 1)
    else:
        step_size = np.maximum(dqam, np.sqrt(r_norm / nt)) * alpha  # (np, 1, 1)

    high, low = 2, 1
    temp = np.concatenate((high * np.ones(samplers // 2), low * np.ones(samplers // 2))).reshape((samplers, 1, 1))
    # delta_beta = 1 / low - 1 / high

    # core
    for t in range(iter):
        v = (np.random.randn(samplers, nt, 1) + 1j * np.random.randn(samplers, nt, 1)) / np.sqrt(2)  # zero-mean, unit-variance
        # construct the proposal
        z_grad = xhat + lr * (grad_preconditioner @ (AH @ r))  # (np, nt, 1)

        z_prop = z_grad + step_size * (covar @ v) # (np, nt, 1)
        x_prop = constellation_norm[np.argmin(abs(z_prop * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
        r_prop = y - A @ x_prop
        r_norm_prop = np.real(np.conj(np.transpose(r_prop, axes=[0, 2, 1])) @ r_prop)  # (np, 1, 1)
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]

        # acceptance step
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (temp)) # (np, 1, 1) noise_var + --> acceptance rate +
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        if index.any():
            xhat[index], r[index], r_norm[index] = x_prop[index], r_prop[index], r_norm_prop[index]
        temp = replica_exchange(temp, r_norm)
        if lr_approx is False and index.any():
            pr_prev = p_mat @ r[index]  # (np_update, nr, 1)
            lr[index] = np.real(np.conj(np.transpose(r[index], axes=[0, 2, 1])) @ pr_prev /
                        (np.conj(np.transpose(pr_prev, axes=[0, 2, 1])) @ pr_prev))  # (np_update, 1, 1)
        if vec_step_size:
            step_size[index] = np.maximum(dqam, abs(AH @ r[index]) / np.sqrt(nt))  # (np_update, nt, 1)
        else:
            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index] / nt)) * alpha  # (np_update, 1, 1)

    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=1))
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse


def replica_exchange(temp, r_norm):
    samplers = temp.shape[0]
    for i in range(samplers // 2):
        delta_h = np.squeeze(r_norm[i] - r_norm[i + samplers // 2])
        delta_beta = np.squeeze(1 / temp[i] - 1 / temp[i + samplers // 2])
        log_pacc = np.minimum(0, delta_beta * delta_h)
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(1, ))
        if p_acc >= p_uni:
            tmp = temp[i].copy()
            temp[i] = temp[i + samplers // 2]
            temp[i + samplers // 2] = tmp
    return temp


def dmhgd(x, A, y, c, noise_var, mu=2, iter=8, samplers=1, lr_approx=False, mmse_init=False, vec_step_size=True,
          net=False, hessian_approx=False, early_stop=False):
    # initialization
    mr, nt = A.shape
    n = mr // c
    yc = y.reshape(c, n, 1)
    Ac = A.reshape(c, n, nt)
    AcH = np.conj(np.transpose(Ac, axes=[0, 2, 1]))  # (c, nt, n)
    if early_stop and noise_var >= 10 ** (- 15.2 / 10) and iter >= 3:
        iter = 3
    if hessian_approx:
        dc = np.transpose(np.sum(np.conj(Ac) * Ac, axis=1, keepdims=True), axes=[0, 2, 1])  # (c, nt, 1)
        AcHAc = dc * np.eye(nt).reshape((1, nt, nt))
    else:
        AcHAc = AcH @ Ac  # (c, nt, nt)
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    es = 1 / dqam
    noise_scale = (2 * dqam) ** 2 / np.log(2)  # change to the neighbor be given a acceptance rate of 50%
    grad_preconditioner = la.inv(np.sum(AcHAc, axis=0) + noise_var / (dqam ** 2) * np.eye(nt))  # (nt, nt)
    AH = np.conj(A).T
    alpha = 0.5 / ((mr / 64) ** (1 / 2))
    # alpha = 1
    ones = np.ones((samplers, nt, 2 ** mu))
    if mr != nt:
        Ainv = la.cholesky(la.inv(np.sum(AcHAc, axis=0)))  # (nt, nt), easy than la.inv(A)
        col_norm = 1 / la.norm(Ainv, axis=1, keepdims=True)
        covar = Ainv * col_norm  # (nt, nt)
        # covar = np.eye(nt)  # choice 5: none
    else:
        Ainv = la.inv(A)  # (nt, nt), easy than la.inv(A)
        col_norm = 1 / la.norm(Ainv, axis=0)  # less effective than row normalization; column normalization tends to eye
        covar = Ainv * col_norm  # (nt, nt)
    # covar = np.eye(nt)
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam

    if lr_approx is False and net is False:  # not scalable
        p_mat = A @ grad_preconditioner @ AH  # (nr, nr)  todo: decentralized -- only can be done by concatenate?
    else:
        p_mat = None

    if mmse_init is True:
        x_mmse = la.inv(AH @ A + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        # xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
        # xhat[np.random.randint(low=0, high=samplers)] = xmmse  # only one sampler initialized by MMSE
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    elif mmse_init == 'cg':
        xinit = np.zeros_like(x)
        x_mmse = cg(xinit, xi=AH @ A + noise_var * np.eye(nt), y=AH @ y, iter=nt)
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                            axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))
    elif mmse_init == 'mf':
        x_c = la.inv(AcHAc) @ (AcH @ yc)  # (c, nt, 1)
        xhat = x_c[c - 1] + grad_preconditioner @ np.sum(AcH @ (yc - Ac @ x_c), axis=0)  # (nt, 1)
        xhat = constellation_norm[np.argmin(abs(xhat * np.ones((nt, 2 ** mu)) - constellation_norm),
                                            axis=1)].reshape(-1, 1)  # quantization
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(nt, samplers))].copy()
    r = y - A @ xhat  # (nr, np)  todo: decentralized
    rc = yc - Ac @ xhat  # (c, n, np)
    r_norm = np.sum(abs(r) ** 2, axis=0)  # (np, )
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()
    if lr_approx is False and net is False:
        pr_prev = p_mat @ r  # (nr, np)
        lr = np.real(np.sum((np.conj(r) * pr_prev), axis=0)) / \
             np.sum(abs(pr_prev) ** 2, axis=0)  # (np, )
    elif net:
        lr = LR
    else:
        lr = 1
    if net:
        dqam = np.sqrt(3 / 2 / (2 ** mu - 1)) * (es / (1 + np.exp(- gamma[0])))
    if vec_step_size:
        step_size = np.maximum(dqam, abs(AH @ r) / np.sqrt(nt))  # (c, nt, 1)
    else:
        step_size = np.maximum(dqam, np.sqrt(r_norm) / nt) * alpha  # (np, )

    acc_rate = 0
    # core
    for t in range(iter):
        # construct the proposal
        if net:
            z_grad = xhat + lr[t] * (grad_preconditioner @ np.sum(AcH @ rc, axis=0))  # (nt, np)
        else:
            z_grad = xhat + lr * (grad_preconditioner @ np.sum(AcH @ rc, axis=0))  # (nt, np)
        # if vec_step_size:
        #     step_size = np.maximum(dqam, abs(y - A @ z_grad) / np.sqrt(nt))  # (c, nt, 1)
        # else:
        #     step_size = np.maximum(dqam, la.norm(y - A @ z_grad, axis=1, keepdims=True) / np.sqrt(nt))  # (c, 1, 1)
        # if t == 0:
        #     step_size = np.maximum(dqam, np.sum(abs(y - A @ z_grad) ** 2, axis=0) / nt)  # (c, 1, 1)
        # v = (np.zeros((nt, samplers)) + 1j * np.zeros((nt, samplers))) / np.sqrt(
        #     2)  # without random noise
        v = (np.random.randn(nt, samplers) + 1j * np.random.randn(nt, samplers)) / np.sqrt(2)  # zero-mean, unit-variance
        z_prop = z_grad + step_size * (covar @ v)  # (nt, np)
        tmp = z_prop.T.reshape((samplers, nt, 1))
        tmp = constellation_norm[np.argmin(abs(tmp * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
        x_prop = tmp.reshape(samplers, nt).T  # (nt, np)
        # x_prop = xhat + lr * grad_preconditioner @ AH @ r
        r_prop = y - A @ x_prop  # (nr, np)
        r_norm_prop = np.sum(abs(r_prop) ** 2, axis=0)  # (np, )
        update = r_norm_survivor > r_norm_prop
        if update.any():
            x_survivor[:, update] = x_prop[:, update]
            r_norm_survivor[update] = r_norm_prop[update]
            if early_stop:
                u, counts = np.unique(r_norm_survivor, return_counts=True)
                if counts[0] > max(samplers // 2, 3) and u[0] == np.amin(r_norm_survivor) and u[
                    0] < mr * noise_var:
                    break

        # acceptance step todo: noise scaling
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (1))  # (np, ) noise_var + --> acceptance rate +
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, ))
        index = p_acc >= p_uni
        acc_rate += sum(index)
        if index.any():
            xhat[:, index], r[:, index], r_norm[index] = x_prop[:, index], r_prop[:, index], r_norm_prop[index]
            rc[:, :, index] = yc - Ac @ xhat[:, index]
        if lr_approx is False and index.any() and net is False:
            pr_prev = p_mat @ r[:, index]  # (nr, np_update)
            lr[index] = np.real(np.sum((np.conj(r[:, index]) * pr_prev), axis=0)) / \
                        np.sum(abs(pr_prev) ** 2, axis=0)  # (np_update, )
        if net:
            dqam = np.sqrt(3 / 2 / (2 ** mu - 1)) * (es / (1 + np.exp(- gamma[t + 1])))
        if vec_step_size:
            step_size = np.maximum(dqam, abs(AH @ r) / np.sqrt(nt))  # (nt, 1)
        else:
            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index]) / nt) * alpha  # (np_update, )

    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=0))
    if 0. not in mse:
        a = None
    x_hat = x_survivor[:, np.argmin(r_norm_survivor)].reshape(-1, 1)

    return x_hat, mse, (acc_rate / samplers / iter), (t + 1)


# Nesterov's accelerated gradient descent
def mhngd_para(x, A, y, noise_var, mu=2, iter=8, samplers=16, ng=8, mmse_init=False, vec_step_size=True,
               adaptive_lr=False, quantize=False, post=8, eps=0.15, sur=False, early_stop=False,
               colored=False, noise_cov=None, constellation_norm=None, restart=False,
               hessian_approx=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    noise_scale = (2 * dqam) ** 2 / np.log(2)  # change to the neighbor be given a acceptance rate of 50%
    es = 1 / dqam
    # constraint the number of iterations for low SNR
    if early_stop and noise_var >= nt / mr * 10 ** (- 15.2 / 10) and iter >= 8:
        iter = 8
    # alpha = 0.5 / ((nt / 64) ** (1 / 2))
    alpha = 1 / ((nt / 8) ** (1 / 3))
    # alpha = 0.5 / (((mr * nt) / (64 * 8)) ** (1 / 2))
    # alpha = 0.5
    AH = np.conj(A).T
    if hessian_approx:
        d = np.sum(np.conj(A) * A, axis=0, keepdims=True)  # (1, nt)
        AHA = d * np.eye(nt)
    else:
        AHA = AH @ A
    if colored:
        Ainv = la.inv(A)
        Ainv = cholesky(Ainv @ noise_cov @ np.conj(Ainv).T, lower=True)
        col_norm = 1 / la.norm(Ainv, axis=1, keepdims=True)
        covar = Ainv * col_norm
    elif mr != nt:
        Ainv = cholesky(la.inv(AHA), lower=True)
        col_norm = 1 / la.norm(Ainv, axis=1, keepdims=True)
        covar = Ainv * col_norm
    else:
        Ainv = la.inv(A)
        col_norm = 1 / la.norm(Ainv, axis=0)
        covar = Ainv * col_norm
    # covar = np.eye(nt)  # todo: use approximated diagonal Hessian

    ones = np.ones((samplers, nt, 2 ** mu))
    zeros = np.zeros((samplers, nt, 1))
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    elif mu == 6:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = constellation_norm * dqam

    if mmse_init:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ (AH @ y)
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
    # x_survivor = xhat.copy()
    nz = nt * samplers
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()
    x_prop_last, change = xhat.copy(), 0

    if colored:
        r_inv = la.inv(noise_cov)
        L = la.norm(AH @ r_inv @ A, 'fro')
    else:
        r_inv = None
        L = la.norm(AHA, 'fro')
        # L = np.amax(la.svd(A, compute_uv=False) ** 2)  # exact L
        # if L >= 8 * np.sqrt(2):
        #     L = None
    lr = 1 / L
    # lr = 0.25  # cannot be too large for naive gradient descent
    if adaptive_lr:
        moment_1, moment_2 = 0, 0
    else:
        moment_1, moment_2 = None, None

    # dqam = np.sqrt(3 / 2 / (2 ** mu - 1)) * (es / (1 + np.exp(- gamma[0])))
    if vec_step_size:  # todo: move to after momentum
        step_size = np.maximum(dqam, abs(r) / np.sqrt(nt))  # (np, nt, 1)
    else:
        step_size = np.maximum(dqam, np.sqrt(r_norm / mr)) * alpha  # (np, 1, 1)

    atm1 = 1  # for momentum factor calculation
    momentum = 0
    beta = 0.9
    # cond = la.cond(AHA)
    # beta = (1 - np.sqrt(1 / cond)) / (1 + np.sqrt(1 / cond))  # beta for strongly convex
    beta_1, beta_2 = 0.9, 0.999  # Exponential decay rates for the moment estimates
    alpha_adam = 0.1  # stepsize for Adam
    epsilon = 1e-8
    avg_ng = 0

    break_iter = False
    # core
    for t in range(iter):
        # construct the proposal
        z_grad = xhat.copy()
        grad_idx = True * np.ones(samplers, dtype=bool)
        # reset momentum and at
        momentum = 0  # no momentum reset is better
        # atm1 = 1  # no at reset is better
        for i in range(ng):  # several GD before a random walk
            # at = (1 + (1 + 4 * atm1 ** 2) ** 0.5) / 2
            # beta = min((atm1 - 1) / at, 0.9)
            # atm1 = at
            avg_ng += sum(grad_idx)
            z_grad_last = z_grad.copy()
            y_grad = z_grad + beta * momentum
            if adaptive_lr:  # todo: NAdam
                g = - AH @ (y - A @ y_grad)
                g = np.concatenate((np.real(g), np.imag(g)), axis=1)
                moment_1 = beta_1 * moment_1 + (1 - beta_1) * g
                moment_2 = beta_2 * moment_2 + (1 - beta_2) * g * g
                moment_1_correct = moment_1 / (1 - beta_1 ** (t+1))
                moment_2_correct = moment_2 / (1 - beta_2 ** (t+1))
                grad_update = - alpha_adam / np.sqrt(t + 1) * moment_1_correct / (np.sqrt(moment_2_correct) + epsilon)
                grad_update = grad_update[:, 0:nt, :] + 1j * grad_update[:, nt:2 * nt, :]
                z_grad = y_grad + grad_update
            else:
                if colored:
                    z_grad = y_grad + lr * (AH @ r_inv @ (y - A @ y_grad))
                else:
                    z_grad = y_grad + lr * (AH @ (y - A @ y_grad))
            momentum = z_grad - z_grad_last  # calculate it after MH correction has similar effect
            if quantize and i != ng - 1 and t >= post:
                x_prop = constellation_norm[np.argmin(abs(z_grad * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
                nz += np.sum(x_prop != x_prop_last)
                # delta = x_prop - x_prop_last
                # r_prop = r - A @ delta
                x_prop_last = x_prop.copy()
                r_prop = y - A @ x_prop
                r_norm_prop = np.real(np.conj(np.transpose(r_prop, axes=[0, 2, 1])) @ r_prop)  # (np, 1, 1)
                update = np.squeeze(r_norm_survivor > r_norm_prop)
                if update.any():
                    update = update & grad_idx
                    x_survivor[update] = x_prop[update]
                    r_norm_survivor[update] = r_norm_prop[update]
                    if sur:
                        z_grad[update] = x_prop[update]
                    if early_stop:
                        u = np.amin(r_norm_survivor)
                        counts = np.sum(r_norm_survivor == u)
                        if counts > max(samplers // 2, 4) and u < 1.5 * mr * noise_var:
                            break_iter = True
                            break
            if restart:
                flag = np.squeeze(- np.conj(np.transpose(AH @ (y - A @ y_grad),axes=[0, 2, 1]))
                                  @ momentum > 0)  # (np, 1, nt) @ (np, nt, 1) --> (np, 1, 1)
                if flag.any():
                    momentum[flag] = zeros[flag]
                    atm1 = 1

        if break_iter:
            break

        # todo: step size combine with learning rate? -- \sqrt(2 lr) preconditioned SGLD
        v = (np.random.randn(samplers, nt, 1) + 1j * np.random.randn(samplers, nt, 1)) / np.sqrt(
            2)  # zero-mean, unit-variance
        z_prop = z_grad + step_size * (covar @ v)
        x_prop = constellation_norm[np.argmin(abs(z_prop * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
        nz += np.sum(x_prop != xhat)
        r_prop = y - A @ x_prop
        r_norm_prop = np.real(np.conj(np.transpose(r_prop, axes=[0, 2, 1])) @ r_prop)  # (np, 1, 1)
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]
            if early_stop:
                u = np.amin(r_norm_survivor)
                counts = np.sum(r_norm_survivor == u)
                if counts > max(samplers // 2, 4) and u < 1.5 * mr * noise_var:
                    break

        # acceptance step
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (1))  # (np, 1, 1)
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        if index.any():
            xhat[index], r[index], r_norm[index] = x_prop[index], r_prop[index], r_norm_prop[index]

        # dqam = np.sqrt(3 / 2 / (2 ** mu - 1)) * (es / (1 + np.exp(- gamma[t + 1])))
        if vec_step_size:
            step_size[index] = np.maximum(dqam, abs(r[index]) / np.sqrt(nt))  # (np_update, nt, 1)
        else:
            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index] / mr)) * alpha # (np_update, 1, 1)

    # # select the sample that minimizes the ML cost
    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=1))
    if 0. not in mse:
        a = None
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse, (avg_ng / samplers / (t + 1)), (t + 1), nz / (samplers + ng * samplers * (t + 1))


def dmhngd(x, A, y, c, noise_var, mu=2, iter=8, samplers=16, ng=8, mmse_init=False, vec_step_size=True,
           quantize=False, post=8, hessian_approx=False, early_stop=False, central_gd=False):
    # initialization
    mr, nt = A.shape
    n = mr // c
    yc = y.reshape(c, n, 1)
    Ac = A.reshape(c, n, nt)
    AcH = np.conj(np.transpose(Ac, axes=[0, 2, 1]))  # (c, nt, n)
    if hessian_approx:
        dc = np.transpose(np.sum(np.conj(Ac) * Ac, axis=1, keepdims=True), axes=[0, 2, 1])  # (c, nt, 1)
        AcHAc = dc * np.eye(nt).reshape((1, nt, nt))
    else:
        AcHAc = AcH @ Ac  # (c, nt, nt)
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    noise_scale = (2 * dqam) ** 2 / np.log(2)  # change to the neighbor be given a acceptance rate of 50%
    # alpha = 0.5 / ((mr / 64) ** (1 / 2))
    alpha = 1
    AH = np.conj(A).T
    AHA = np.sum(AcHAc, axis=0)
    if mr != nt:
        Ainv = la.cholesky(la.inv(AHA))  # (nt, nt), easy than la.inv(A)
    else:
        Ainv = la.inv(A)
    col_norm = 1 / la.norm(Ainv, axis=0)
    covar = Ainv * col_norm
    # covar = np.eye(nt)  # todo: use approximated diagonal Hessian

    ones = np.ones((samplers, nt, 2 ** mu))
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam

    if mmse_init:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ (AH @ y)
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(nt, samplers))].copy()
    # x_survivor = xhat.copy()
    nz = nt * samplers
    r = y - A @ xhat  # (nr, np)
    rc = yc - Ac @ xhat  # (c, n, np)
    r_norm = np.sum(abs(r) ** 2, axis=0)  # (np, )
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()
    x_prop_last, change = xhat.copy(), 0

    if central_gd:
        L = la.norm(AHA, 'fro')
    else:
        L = la.norm(AcHAc, 'fro', axis=(1, 2))  # (c, )
    lr = 1 / L  # (c,)
    # lr = 0.25  # cannot be too large for naive gradient descent
    if vec_step_size:  # todo: move to after momentum
        step_size = np.maximum(dqam, abs(r) / np.sqrt(nt))  # (np, nt, 1)
    else:
        step_size = np.maximum(dqam, np.sqrt(r_norm) / nt) * alpha  # (np, )

    atm1 = 1  # for momentum factor calculation
    momentum = 0
    beta = 0.9
    beta_1, beta_2 = 0.9, 0.999  # Exponential decay rates for the moment estimates
    alpha_adam = 0.1  # stepsize for Adam
    epsilon = 1e-8
    avg_ng = 0

    break_iter = False
    # core
    for t in range(iter):
        # construct the proposal
        z_grad = xhat.copy()  # (nt, np)
        grad_idx = True * np.ones(samplers, dtype=bool)

        if central_gd:
            for i in range(ng):
                # at = (1 + (1 + 4 * atm1 ** 2) ** 0.5) / 2
                # beta = (atm1 - 1) / at
                # atm1 = at
                avg_ng += sum(grad_idx)
                z_grad_last = z_grad.copy()
                y_grad = z_grad + beta * momentum  # (nt, np)
                z_grad = y_grad + lr * (AH @ (y - A @ y_grad))  # (nt, np)
                momentum = z_grad - z_grad_last
                if quantize and i != ng - 1 and t >= post:
                    tmp = z_grad.T.reshape((samplers, nt, 1))
                    tmp = constellation_norm[np.argmin(abs(tmp * ones - constellation_norm),
                                                       axis=2)].reshape(-1, nt, 1)  # quantization
                    x_prop = tmp.reshape(samplers, nt).T  # (nt, np)
                    nz += np.sum(x_prop != x_prop_last)
                    x_prop_last = x_prop.copy()
                    r_prop = y - A @ x_prop
                    r_norm_prop = np.sum(abs(r_prop) ** 2, axis=0)  # (np, )
                    update = (r_norm_survivor > r_norm_prop)
                    if update.any():
                        x_survivor[:, update] = x_prop[:, update]
                        r_norm_survivor[update] = r_norm_prop[update]
                        if early_stop:
                            u, counts = np.unique(r_norm_survivor, return_counts=True)
                            if counts[0] > max(samplers // 2, 4) and u[0] == np.amin(r_norm_survivor) and u[
                                0] < mr * noise_var:
                                break_iter = True
                                break
        else:
            for k in range(c):
                # reset momentum and at
                # if k != 0:
                momentum = 0  # no momentum reset is better
                atm1 = 1  # at reset is better
                # L = la.norm(AcH[k] @ Ac[k], 'fro')
                # lr = 1 / L
                for i in range(ng // c):  # several GD before a random walk
                    at = (1 + (1 + 4 * atm1 ** 2) ** 0.5) / 2
                    beta = (atm1 - 1) / at
                    atm1 = at
                    avg_ng += sum(grad_idx)
                    z_grad_last = z_grad.copy()
                    y_grad = z_grad + beta * momentum
                    z_grad = y_grad + lr[k] * (AcH[k] @ (yc[k] - Ac[k] @ y_grad))  # todo: lr for each cluster
                    momentum = z_grad - z_grad_last  # calculate it after MH correction has similar effect
                    if quantize and k == c - 1 and t >= post:
                        tmp = z_grad.T.reshape((samplers, nt, 1))
                        tmp = constellation_norm[np.argmin(abs(tmp * ones - constellation_norm),
                                                           axis=2)].reshape(-1, nt, 1)  # quantization
                        x_prop = tmp.reshape(samplers, nt).T  # (nt, np)
                        nz += np.sum(x_prop != x_prop_last)
                        x_prop_last = x_prop.copy()
                        r_prop = y - A @ x_prop
                        r_norm_prop = np.sum(abs(r_prop) ** 2, axis=0)  # (np, )
                        update = (r_norm_survivor > r_norm_prop)
                        if update.any():
                            x_survivor[:, update] = x_prop[:, update]
                            r_norm_survivor[update] = r_norm_prop[update]
                            if early_stop:
                                u, counts = np.unique(r_norm_survivor, return_counts=True)
                                if counts[0] > max(samplers // 2, 4) and u[0] == np.amin(r_norm_survivor) and u[0] < mr * noise_var:
                                    break_iter = True
                                    break

        if break_iter:
            break

        # todo: step size combine with learning rate? -- \sqrt(2 lr) preconditioned SGLD
        v = (np.random.randn(nt, samplers) + 1j * np.random.randn(nt, samplers)) / np.sqrt(
            2)  # zero-mean, unit-variance
        z_prop = z_grad + step_size * (covar @ v)  # (nt, np)
        tmp = z_prop.T.reshape((samplers, nt, 1))
        tmp = constellation_norm[np.argmin(abs(tmp * ones - constellation_norm),
                                           axis=2)].reshape(-1, nt, 1)  # quantization
        x_prop = tmp.reshape(samplers, nt).T  # (nt, np)
        nz += np.sum(x_prop != xhat)
        r_prop = y - A @ x_prop  # (nr, np)
        r_norm_prop = np.sum(abs(r_prop) ** 2, axis=0)  # (np, )
        update = r_norm_survivor > r_norm_prop
        if update.any():
            x_survivor[:, update] = x_prop[:, update]
            r_norm_survivor[update] = r_norm_prop[update]
            if early_stop:
                u, counts = np.unique(r_norm_survivor, return_counts=True)
                if counts[0] > max(samplers // 2, 4) and u[0] == np.amin(r_norm_survivor) and u[0] < mr * noise_var:
                    break

        # acceptance step
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (1))  # (np, )
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, ))
        index = p_acc >= p_uni
        if index.any():
            xhat[:, index], r[:, index], r_norm[index] = x_prop[:, index], r_prop[:, index], r_norm_prop[index]
            rc[:, :, index] = yc - Ac @ xhat[:, index]
        if vec_step_size:
            step_size[index] = np.maximum(dqam, abs(r[index]) / np.sqrt(nt))  # (np_update, nt, 1)
        else:
            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index]) / nt) * alpha # (np_update, )

    # # select the sample that minimizes the ML cost
    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=0))
    if 0. not in mse:
        a = None
    x_hat = x_survivor[:, np.argmin(r_norm_survivor)].reshape(-1, 1)

    return x_hat, mse, (avg_ng / samplers / (t + 1)), (t + 1), nz / (samplers + ng * samplers * (t + 1))


def mala_para(x, A, y, noise_var, mu=2, iter=8, samplers=16, mmse_init=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    noise_scale = (2 * dqam) ** 2 / np.log(2)  # change to the neighbor be given a acceptance rate of 50%
    AH = np.conj(A).T
    AHA = AH @ A
    ones = np.ones((samplers, nt, 2 ** mu))
    gamma = 1
    if mr != nt:
        # Ainv = (cholesky(la.inv(AHA), lower=True)) ** (1/4)  # choice 1: lower triangular matrix
        # Ainv = la.inv(AHA)  # choice 2: inv(AHA)
        # Ainv = sqrtm(Ainv)  # choice 4: sqrtm(inv(AHA))
        # col_norm = 1 / la.norm(Ainv, axis=0)
        # covar = Ainv * col_norm
        # covar = np.sqrt(covar)  # choice 3: sqrt(norm(inv(AHA)))
        covar = np.eye(nt)  # choice 5: none
    else:
        # Ainv = la.inv(A)
        # col_norm = 1 / la.norm(Ainv, axis=0)
        # covar = Ainv * col_norm
        covar = np.eye(nt)  # choice 5: none
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam

    if mmse_init is True:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        # xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
        # xhat[np.random.randint(low=0, high=samplers)] = xmmse  # only one sampler initialized by MMSE
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    elif mmse_init == 'cg':
        xinit = np.zeros_like(x)
        x_mmse = cg(xinit, xi=AHA + noise_var * np.eye(nt), y=AH @ y, iter=nt // 2)
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                            axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()

    L = la.norm(AHA, 'fro')
    lr = 1 / L
    # lr = 0.25  # cannot be too large for naive gradient descent

    alpha, eps = 0.99, 1e-5
    momentum = 0
    # core
    for t in range(iter):
        v = (np.random.randn(samplers, nt, 1) + 1j * np.random.randn(samplers, nt, 1)) / np.sqrt(2)  # zero-mean, unit-variance
        # construct the proposal -- use underdamped Langevin
        grad = AH @ r
        z_prop = xhat + lr * 4 / gamma * covar @ momentum
        momentum_half = momentum + lr * grad

        momentum = alpha * momentum + (1 - alpha) * grad * grad
        preconditioner = 1 / (np.sqrt(abs(momentum)) + eps) * np.eye(nt)

        z_grad = xhat - lr * preconditioner @ grad  # (np, nt, 1)
        # step_size = np.maximum(np.sqrt(2 * lr * preconditioner / nt), dqam * np.eye(nt))
        step_size = np.sqrt(2 * lr * preconditioner / nt)  # todo
        z_prop = z_grad + step_size @ v  # (np, nt, 1)
        x_prop = constellation_norm[np.argmin(abs(z_prop * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
        # x_prop = xhat + lr * grad_preconditioner @ AH @ r
        r_prop = y - A @ x_prop
        r_norm_prop = np.real(np.conj(np.transpose(r_prop, axes=[0, 2, 1])) @ r_prop)  # (np, 1, 1)
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]

        # acceptance step
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (noise_var))  # (np, 1, 1)
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        if index.any():
            xhat[index], r[index], r_norm[index] = x_prop[index], r_prop[index], r_norm_prop[index]

    # mse = np.squeeze(np.mean(abs(xhat - x) ** 2, axis=1))
    # # select the sample that minimizes the ML cost
    # x_hat = xhat[np.argmin(r_norm), :, :].copy()
    mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=1))
    # r = y - A @ x_survivor
    # r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse