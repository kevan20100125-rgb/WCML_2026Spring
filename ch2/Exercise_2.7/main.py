import numpy as np
import os

import tensorflow.compat.v1 as tf
import scipy.io as sio
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

# import our problems, networks and training modules
from tools import networks, train, raputil

K = 64
mu = 2                  # QPSK pilots for the 1st OFDM symbol
data_mu = 6             # 64-QAM data for the 2nd OFDM symbol
SNR_train = [5, 10, 15, 20, 25, 30, 35, 40]

# you can tune these two if needed
training_epochs = 800
batch_size = 512

ce_type = 'dnn'  # channel estimation: 'mmse', 'dnn'
test_ce = False
CP_flag = True
NoCP = (CP_flag is False)

BER = []
prob = []
x_hat_T = []
sess, input_holder, output = None, None, None
MSE_T, MSE_F = [], []

for i in range(0, len(SNR_train)):
    print("\nSNR=", SNR_train[i])

    sess, input_holder, output = None, None, None

    if ce_type == 'dnn':
        sess, input_holder, output = networks.build_ce_dnn(
            K,
            SNR_train[i],
            training_epochs=training_epochs,
            batch_size=batch_size,
            savefile='dnn_ce/CE_DNN_' + ('CPFREE_' if CP_flag is False else '') +
                     str(2 ** data_mu) + 'QAM_SNR_' + str(SNR_train[i]) + 'dB.npz',
            test_flag=test_ce,
            cp_flag=CP_flag,
            no_cp=NoCP,
            data_mu=data_mu,
            nh1=500,
            nh2=250,
        )

    if test_ce:
        mse_t, mse_f = raputil.test_ce(
            sess,
            input_holder,
            output,
            SNR_train[i],
            est_type=ce_type,
            NoCP=NoCP,
            CP_flag=CP_flag,
            data_mu=data_mu,
        )
        MSE_T.append(mse_t)
        MSE_F.append(mse_f)

    if sess is not None:
        sess.close()
    tf.reset_default_graph()

print('BER', BER)
BER_matlab = np.array(BER)
print('MSE_T', MSE_T)
print('MSE_F', MSE_F)

savefile = 'MSE_' + ce_type + '_' + str(2 ** data_mu) + 'QAM' + ('_CP_FREE' if CP_flag is False else '')
if test_ce:
    sio.savemat(
        savefile + '.mat',
        {
            savefile: np.asarray(MSE_F, dtype=np.float64),
            'snr_db': np.asarray(SNR_train, dtype=np.float64),
        }
    )