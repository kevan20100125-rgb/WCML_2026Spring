import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_curve(mat_path):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError('Cannot find {}'.format(mat_path))

    mat = sio.loadmat(mat_path)
    snr_db = np.asarray(mat['snr_db']).reshape(-1)

    curve = None
    for k, v in mat.items():
        if k.startswith('__') or k == 'snr_db':
            continue
        curve = np.asarray(v).reshape(-1)
        break

    if curve is None:
        raise RuntimeError('No valid curve variable found in {}'.format(mat_path))

    curve_db = 10 * np.log10(np.maximum(curve, 1e-12))
    return snr_db, curve_db


snr1, mmse_cp = load_curve('MSE_mmse_64QAM.mat')
snr2, dnn_cp = load_curve('MSE_dnn_64QAM.mat')
snr3, mmse_no_cp = load_curve('MSE_mmse_64QAM_CP_FREE.mat')
snr4, dnn_no_cp = load_curve('MSE_dnn_64QAM_CP_FREE.mat')

plt.figure(figsize=(7.2, 5.2))

plt.plot(snr1, mmse_cp, 'o-', color='tab:blue', linewidth=1.8, markersize=7,
         markerfacecolor='tab:blue', markeredgewidth=1.0, label='LMMSE')

plt.plot(snr2, dnn_cp, '^-', color='tab:orange', linewidth=1.8, markersize=7,
         markerfacecolor='tab:orange', markeredgewidth=1.0, label='DNN')

plt.plot(snr3, mmse_no_cp, 'o--', color='tab:green', linewidth=1.6, markersize=7,
         markerfacecolor='tab:green', markeredgewidth=1.0, dashes=(4, 3), label='LMMSE w/o CP')

plt.plot(snr4, dnn_no_cp, '^--', color='tab:red', linewidth=1.6, markersize=7,
         markerfacecolor='tab:red', markeredgewidth=1.0, dashes=(4, 3), label='DNN w/o CP')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('MSE(dB)', fontsize=13)
plt.xlim(5, 40)
plt.ylim(-45, -10)
plt.xticks(np.arange(5, 41, 5))
plt.grid(True, alpha=0.25)
plt.legend(loc='lower left', framealpha=1.0, fancybox=False, edgecolor='0.6')
plt.tight_layout()

plt.savefig('Figure_2_9_reproduced.png', dpi=200)
plt.savefig('Figure_2_9_reproduced.pdf')
print('Saved: Figure_2_9_reproduced.png')
print('Saved: Figure_2_9_reproduced.pdf')