
import sys
import numpy as np
import numpy.linalg as la
from tools.utils import QAM_Modulation, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation,\
    QAM_Demodulation
from tools.MIMO_detection import corr_channel
import scipy.io as sio


np.random.seed(1)  # numpy is good about making repeatable output
mr, nt = 8, 8
mu = 4
snr = 15
samples = 32
samplers = 16
num_trail = 10000
detect_type = 'MHGD_PARA'  #
lr_approx = False
mmse_init = True
vec_step_size = False
d_tune = True
adaptive_lr = True
r_norm = 0
channel_type = ''
rho_tx, rho_rx = 0.6, 0.6

if channel_type == 'corr':
    sqrtRtx, sqrtRrx = corr_channel(mr, nt, rho_tx=rho_tx, rho_rx=rho_rx)
else:
    sqrtRtx, sqrtRrx = None, None

err_bits = np.zeros(samples, dtype=int)
i = 0
denom, nom = 0, 0
for i in range(num_trail):
    bits = np.random.binomial(n=1, p=0.5, size=(nt * mu, ))
    bits_mod = QAM_Modulation(bits, mu)
    x = bits_mod.reshape(nt, 1)

    H = np.sqrt(1 / 2 / mr) * (np.random.randn(mr, nt) + 1j * np.random.randn(mr, nt))
    if channel_type == 'corr':  # Correlated MIMO channel
        H = sqrtRrx @ H @ sqrtRtx
    y = H @ x

    nom += np.real(np.conj(y.T) @ y)
    denom += np.real(np.conj(x.T) @ x)

print('gain:', nom / denom)
