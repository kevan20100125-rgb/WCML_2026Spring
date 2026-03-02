import numpy as np
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation


def sgld(x, A, y, noise_var, mu=2, iter=8, samplers=16, grad_iter=200):  # only consider QPSK in the current version
    # initialization
    mr, nt = A.shape
    AH = np.conj(A).T
    M = 2 ** mu
    dqam = np.sqrt(3 / 2 / (M - 1))  # Eavg = 2/3*(M-1)
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
        theta = np.pi / 4  # initial phase
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam
    alpha, beta, eta, xi, delta = 0.07, 1.1, 0.05, 10, 1e-4
    D = 4 * np.sqrt(2 * eta * nt / xi)

    xhat_real = np.random.uniform(low=0.0, high=1.0, size=(samplers, nt, 1))
    xhat_imag = np.sqrt(1 - xhat_real ** 2) * (2 * np.random.binomial(n=1, p=0.5, size=(samplers, nt, 1)) - 1)
    xhat = xhat_real + 1j * xhat_imag  # (np, nt, 1)
    x_list = np.zeros((samplers, nt, iter), dtype=complex)

    for t in range(iter):
        w = (np.random.randn(samplers, nt, 1) + 1j * np.random.randn(samplers, nt, 1)) / np.sqrt(2)
        grad = 2 * AH @ (A @ xhat - y) + 2 * M * alpha * (xhat ** M - np.exp(1j * M * theta)) \
               * np.conj(xhat) ** (M - 1) + 4 * beta * xhat * (abs(xhat) ** 2 - 1)
        z = xhat - eta * grad + np.sqrt(2 * eta / xi) * w
        r = z - xhat
        d = np.sqrt(np.real(np.conj(np.transpose(r, axes=[0, 2, 1])) @ r))  # (np, 1, 1)
        idx = np.squeeze(d <= D, axis=(1, 2))  # (np, )
        if idx.any():
            xhat[idx] = z[idx]
        x_list[:, :, t:t + 1] = xhat
    res = y - A @ x_list  # (np, nr, iter)
    res_norm = np.sum(abs(res) ** 2, axis=1)  # (np, iter)
    idx = np.argmin(res_norm, axis=1)  # (np,)
    xhat = x_list[np.arange(samplers), :, idx][:, :, np.newaxis]  # (np, nt, 1)

    for g in range(grad_iter):
        grad = 2 * AH @ (A @ xhat - y) + 2 * M * alpha * (xhat ** M - np.exp(1j * M * theta)) \
               * np.conj(xhat) ** (M - 1) + 4 * beta * xhat * (abs(xhat) ** 2 - 1)  # (np, nt, 1)
        xhat = xhat - eta * grad
        idx = np.squeeze(np.real(np.conj(np.transpose(grad, axes=[0, 2, 1])) @ grad) < delta, axis=(1, 2))
        if idx.all():
            break
    # quantization
    shat = constellation_norm[np.argmin(abs(xhat * np.ones((1, 1, M)) - constellation_norm),
                                        axis=2)].reshape(-1, nt, 1)
    mse = np.squeeze(np.mean(abs(x - shat) ** 2, axis=1))
    if 0. not in mse:
        a = None
    res = y - A @ shat  # (np, nr, 1)
    res_norm = np.real(np.conj(np.transpose(res, axes=[0, 2, 1])) @ res)  # (np, 1, 1)
    shat = shat[np.argmin(res_norm), :, :]

    return shat, mse
