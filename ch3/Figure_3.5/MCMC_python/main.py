#!/usr/bin/python
from __future__ import division
from __future__ import print_function
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import numpy as np
import scipy.io as sio
from tools import MIMO_detection

np.random.seed(1)  # numpy is good about making repeatable output

snr_list = np.arange(0, 45, 5)  # SNR list

BER = []
SER = []
FER = []
BLER = []
other_stats = []
SNR_test = []


class SysIn(object):
    mu = 4  # modulation order:2^mu QPSK:mu=2 16QAM:mu=4
    Mr = 8  # number of receiving antennas
    Nt = 8  # number of transmitting antennas
    T = 10  # number of EP iterations
    samples = 8  # number of samples for MCMC
    samplers = 4  # number of samplers for MCMC
    detect_type = 'MHGD'  # 'EP', 'MMSE', 'MHGD', 'Gibbs', 'NAG_MCMC', 'ML'
    lr_approx = False
    mmse_init = True
    ng = 1  # number of successive NAG iterations for NAG-MCMC
    quantize, post, sur = False, 0, False  # sample augmentation for NAG-MCMC
    es = False  # early stopping for NAG-MCMC
    channel_type = 'rayleigh'  # channel type: 'rayleigh', 'corr'
    rho_tx, rho_rx = 0.7, 0.7
    csi = 0  # 0: perfect csi; 1: add awgn; 2: channel estimation
    mcmc_dict = {'MHGD', 'Gibbs', 'NAG_MCMC'}
    savefile = 'Results_' + detect_type + '_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM' + \
               (('_T' + str(T)) if detect_type == 'EP_real_v3' else '') + \
               (('_' + channel_type + str(rho_tx).replace('.', '') + '_' + str(rho_rx).replace('.', '')) if channel_type == 'corr' else '') + \
               ('_ce' if csi == 2 else '') + \
               (('_np' + str(samplers) + '_ns' + str(samples)) if detect_type in mcmc_dict else '') + \
               (('_ng' + str(ng)) if detect_type == 'NAG_MCMC' else '') + \
               ('_lr_approx' if (detect_type in mcmc_dict ) and lr_approx else '') + \
               ('_mmse' if (detect_type in mcmc_dict) and mmse_init is True else '') + \
               (('_quan' + str(post)) if (detect_type == 'NAG_MCMC') and quantize else '') + \
               ('_es' if (detect_type == 'NAG_MCMC') and es else '') + \
               ('_sur' if (detect_type == 'NAG_MCMC') and sur else '') + ''

sysIn = SysIn()

for i in range(5, 6):
    print("SNR=", snr_list[i])
    SNR_test.append(snr_list[i])
    np.random.seed(0)
    ber, ser, stats = MIMO_detection.MIMO_detection_simulate(sysIn, snr_list[i])
    SER.append(ser)
    BER.append(ber)
    other_stats.append(stats)

print('BER', BER)
print('SER', SER)
results = np.array([BER, SER])
other_stats = np.array(other_stats).T
SNR_test = np.array(SNR_test)

sio.savemat(sysIn.savefile + '.mat', {sysIn.savefile: results, 'other_stats': other_stats, 'SNR': SNR_test})
