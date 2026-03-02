
import numpy as np
import numpy.linalg as la
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
from scipy.linalg import cholesky, sqrtm
from numba import jit


# @jit()
def mhgd():  # todo

    return


def nag_mcmc(x, A, y, noise_var, mu=2, iter=8, samplers=16, ng=8, mmse_init=False,
             quantize=False, post=8, sur=False, early_stop=False, constellation_norm=None):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AH = np.conj(A).T
    AHA = AH @ A
    # alpha = 0.5 / ((nt / 64) ** (1 / 2))
    alpha = 1 / ((nt / 8) ** (1 / 3))  # scaling parameter
    ones = np.ones((samplers, nt, 2 ** mu))

    # generate random walk covariance
    if mr != nt:
        Ainv = cholesky(la.inv(AHA), lower=True)
        col_norm = 1 / la.norm(Ainv, axis=1, keepdims=True)
        covar = Ainv * col_norm
    else:
        Ainv = la.inv(A)
        col_norm = 1 / la.norm(Ainv, axis=0)
        covar = Ainv * col_norm
    # covar = np.eye(nt)

    # constellation
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    elif mu == 6:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = constellation_norm * dqam

    # initialize the estimate with size (np, nt, 1), np is for parallel samplers
    if mmse_init:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ (AH @ y)
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** mu)) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # residual: (np, nr, 1)
    r_norm = np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r)  # residual norm: (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()  # survivor sample and its residual norm

    # upper bound of L-smoothness parameter & learning rate
    L = la.norm(AHA, 'fro')
    lr = 1 / L

    step_size = np.maximum(dqam, np.sqrt(r_norm / mr)) * alpha  # (np, 1, 1)

    atm1 = 1  # for momentum factor calculation
    momentum = 0
    beta = 0.9

    break_iter = False
    # core
    for t in range(iter):
        # construct the proposal
        z_grad = xhat.copy()
        grad_idx = True * np.ones(samplers, dtype=bool)
        # reset momentum and at
        # momentum = 0  # no momentum reset is better
        # atm1 = 1  # no at reset is better

        # multiple GDs per random walk
        for i in range(ng):
            # at = (1 + (1 + 4 * atm1 ** 2) ** 0.5) / 2
            # beta = min((atm1 - 1) / at, 0.9)
            # atm1 = at
            z_grad_last = z_grad.copy()
            y_grad = z_grad + beta * momentum
            z_grad = y_grad + lr * (AH @ (y - A @ y_grad))
            momentum = z_grad - z_grad_last  # calculate it after MH correction has similar effect
            if quantize and i != ng - 1 and t >= post:
                x_prop = constellation_norm[np.argmin(abs(z_grad * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
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

        if break_iter:
            break

        v = (np.random.randn(samplers, nt, 1) + 1j * np.random.randn(samplers, nt, 1)) / np.sqrt(
            2)  # zero-mean, unit-variance
        z_prop = z_grad + step_size * (covar @ v)
        x_prop = constellation_norm[np.argmin(abs(z_prop * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
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

        if t != iter - 1:
            # acceptance step
            log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (1))  # (np, 1, 1)
            p_acc = np.exp(log_pacc)
            p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
            index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
            if index.any():
                xhat[index], r[index], r_norm[index] = x_prop[index], r_prop[index], r_norm_prop[index]

            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index] / mr)) * alpha # (np_update, 1, 1)

    # select the sample that minimizes the ML cost
    # mse = np.squeeze(np.mean(abs(x - x_survivor) ** 2, axis=1))
    mse = 0.
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse, (t + 1)

