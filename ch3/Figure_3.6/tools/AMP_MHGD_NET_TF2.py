#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train_tf2 import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
from .AMP_NN_TF2 import AMP_NN
from .MHGD_NET_TF2 import MHGDNet
import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio


def train_amp_mhgd_net(test=False, trainset = None):
    Mr, Nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size

    savefile1 = trainset.savefile1
    directory = './model/' + 'AMP_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile1 = directory + '/' + savefile1
    model1 = AMP_NN(trainset=trainset)

    savefile2 = trainset.savefile2
    directory = './model/' + 'MHGD_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile2 = directory + '/' + savefile2
    model2 = MHGDNet(trainset=trainset)

    savefile = trainset.savefile
    directory = './model/' + 'AMP_MHGD_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    # mkdir(directory)
    savefile = directory + '/' + savefile

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, lr_decay, name='lr')
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    @tf.function
    def grad(x, y, H, sigma2, bs, label):
        with tf.GradientTape() as tape:
            x_amp, loss, mse = model1(x, y, H, sigma2, bs=bs, test=test, label=label)
            xhat, loss, mse = model2(x, y, H, sigma2, bs=bs, test=test, x_fs=x_amp)
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model1.trainable_variables), trainset.grad_clip)
        return loss, grads

    # generate validation set
    _, _, _, _, _, yval, xval, Hval, sigma2val, labelval = sample_gen(trainset, 1, vsample_size, label=True)
    labelval = labelval.astype(np.float64)
    # sigma2val = Nt / Mr * 10 ** (-SNR / 10)
    val_batch_size = vsample_size // total_batch
    # call to generate trainable variables
    xbatch = xval[0 * val_batch_size: (0 + 1) * val_batch_size]
    ybatch = yval[0 * val_batch_size: (0 + 1) * val_batch_size]
    Hbatch = Hval[0 * val_batch_size:(0 + 1) * val_batch_size]
    sigma2batch = sigma2val[0 * val_batch_size:(0 + 1) * val_batch_size]
    labelbatch = labelval[0 * val_batch_size:(0 + 1) * val_batch_size]
    _, _ = grad(xbatch, ybatch, Hbatch, sigma2batch, val_batch_size,
                labelbatch)

    state = load_trainable_vars(model1, savefile1)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)
    state = load_trainable_vars(model2, savefile2)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)

    if test:
        return (model1, model2)

    loss_history = []
    save = {}  # for the best model
    ivl = 5
    # generate validation set
    _, _, _, _, yval, xval, Hval, sigma2val = sample_gen(trainset, 1, vsample_size)
    # sigma2val = Nt / Mr * 10 ** (-SNR / 10)
    val_batch_size = vsample_size // total_batch

    for i in range(maxit + 1):
        if i % ivl == 0:  # validation:don't use optimizer
            loss = 0.
            for m in range(total_batch):
                xbatch = xval[m * val_batch_size: (m + 1) * val_batch_size]
                ybatch = yval[m * val_batch_size: (m + 1) * val_batch_size]
                Hbatch = Hval[m * val_batch_size:(m + 1) * val_batch_size]
                sigma2batch = sigma2val[m * val_batch_size:(m + 1) * val_batch_size]
                x_amp, loss_batch, mse = model1(xbatch, ybatch, Hbatch, sigma2batch,
                                                label=labelbatch, bs=val_batch_size, test=test)
                xhat, loss_batch, mse = model2(xbatch, ybatch, Hbatch, sigma2batch,
                                               bs=val_batch_size, test=test, x_fs=x_amp)
                loss += loss_batch
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)
                    #
            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'
                             .format(i=i, loss=loss, best=loss_best))
            sys.stdout.flush()
            if i % (100 * 1) == 0:
                print('')

        # generate trainset
        y, x, H, sigma2, _, _, _, _ = sample_gen(trainset, batch_size * total_batch, 1)
        # sigma2 = Nt / Mr * 10 ** (-SNR / 10)
        for m in range(total_batch):
            train_loss, _, grad = sess.run((loss_, train, grads_),
                                           feed_dict={y_: y[m * batch_size:(m + 1) * batch_size],
                                                      x_: x[m * batch_size:(m + 1) * batch_size],
                                                      H_: H[m * batch_size:(m + 1) * batch_size],
                                                      sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                                      bs_: batch_size})
            # if grad > 100.0:
            #     pass

    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k)
            # print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin() * ivl)

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    para = {}
    for k, v in np.load(savefile).items():
        para[k.replace(':', '').replace('/', '_')] = v
    sio.savemat(savefile.replace('.npz', '.mat'), para)

    return