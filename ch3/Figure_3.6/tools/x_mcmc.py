
import numpy as np
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation


def x_mcmc(x, H, y, noise_var, mu=2, iter=320, samplers=50):
    mr, nt = H.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)

    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) * dqam
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) * dqam
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) * dqam

    base = 2 ** np.flip(np.arange(mu))
    b = np.random.randint(low=0, high=2, size=(samplers * nt, mu))
    xhat = constellation_norm[np.sum(b * base, axis=1, keepdims=True)]  # (samplers*nt, 1)
    xhat = np.reshape(xhat, (nt, samplers))
    xhatp = xhat.copy()
    xhatm = xhat.copy()

    x_list = np.zeros((nt, iter * samplers), dtype=complex)
    r_norm_list = np.zeros((1, iter * samplers))

    for i in range(iter):

        for k in range(nt * mu):
            col = k % mu
            row = np.floor(k / mu).astype(int)
            b[row::nt, col] = 1
            xhatp[row, :] = constellation_norm[np.sum(b[row::nt, :] * base, axis=1, keepdims=True)].reshape(-1)
            b[row::nt, col] = 0
            xhatm[row, :] = constellation_norm[np.sum(b[row::nt, :] * base, axis=1, keepdims=True)].reshape(-1)
            rp = y - H @ xhatp
            rm = y - H @ xhatm
            dp = np.real(np.sum(np.conj(rp) * rp, axis=0))  # (samplers,)
            dm = np.real(np.sum(np.conj(rm) * rm, axis=0))  # (samplers,)
            gamma = (dm - dp) / np.minimum(dp, dm) * nt
            p_gibbs = 1 / (1 + np.exp(- gamma))
            p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers,))
            index = p_gibbs > p_uni
            tmp = np.transpose(np.reshape(b, (samplers, nt, mu)), (1, 2, 0))  # (nt, mu, samplers)
            tmp[row, col, index] = 1
            tmp[row, col, ~index] = 0
            b = np.reshape(np.transpose(tmp, (2, 0, 1)), (samplers * nt, mu))
            xhat[row, index] = xhatp[row, index]
            xhat[row, ~index] = xhatm[row, ~index]
            xhatp = xhat.copy()
            xhatm = xhat.copy()

        x_list[:, i * samplers:(i + 1) * samplers] = xhat
        r = y - H @ xhat
        r_norm_list[:, i * samplers:(i + 1) * samplers] = np.real(np.sum(np.conj(r) * r, axis=0)).reshape(1, -1)

    min_idx = np.argmin(r_norm_list)
    xhat = x_list[:, min_idx]

    return xhat
