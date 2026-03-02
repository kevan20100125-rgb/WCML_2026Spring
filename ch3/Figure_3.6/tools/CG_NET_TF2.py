#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train_tf2 import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio

from tensorflow.keras import Model

# todo: 1. CG-NET scalar  2. CG-NET vector 3. CG-NET quantized; reference to FS-NET


def train_cg_net(test=False, trainset=None):
    Mr, Nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size
    savefile = trainset.savefile
    directory = './model/' + 'CG_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    model = CGNet(trainset=trainset)

    state = load_trainable_vars(model, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)

    if test:
        return model

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, lr_decay, name='lr')
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    @tf.function
    def grad(x, y, H, sigma2, bs):
        with tf.GradientTape() as tape:
            xhat, loss, mse = model(x, y, H, sigma2, bs=bs, test=test)
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_variables), trainset.grad_clip)
        return loss, grads

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
                xhat, loss_batch, mse = model(xbatch, ybatch, Hbatch, sigma2batch, bs=val_batch_size, test=test)
                loss += loss_batch
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            if loss == loss_best:
                for v in model.trainable_variables:
                    save[str(v.name)] = v.numpy()
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
            xbatch = x[m * batch_size:(m + 1) * batch_size]
            ybatch = y[m * batch_size:(m + 1) * batch_size]
            Hbatch = H[m * batch_size:(m + 1) * batch_size]
            sigma2batch = sigma2[m * batch_size:(m + 1) * batch_size]
            loss_batch, grads = grad(xbatch, ybatch, Hbatch, sigma2batch, batch_size)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # if grad > 100.0:
            #     pass

    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in model.trainable_variables])
    for k, d in save.items():
        if k in tv:
            tv[k].assign(tf.convert_to_tensor(d))
            print('restoring ' + k)
            # print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin() * ivl)

    state['done'] = done
    state['log'] = log
    save_trainable_vars(model, savefile, **state)

    para = {}
    for k, v in np.load(savefile).items():
        para[k.replace(':', '').replace('/', '_')] = v
    sio.savemat(savefile.replace('.npz', '.mat'), para)

    return


class CGNet(Model):
    def __init__(self, trainset=None):
        super(CGNet, self).__init__()
        self.mr, self.nt, self.mu = trainset.m, trainset.n, trainset.mu
        self.dqam = np.sqrt(3 / 2 / (2 ** self.mu - 1))
        self.es = 1 / self.dqam
        if self.mu == 2:
            self.constellation_norm = _QPSK_Constellation.reshape(-1) * self.dqam
        elif self.mu == 4:
            self.constellation_norm = _16QAM_Constellation.reshape(-1) * self.dqam
        else:
            self.constellation_norm = _64QAM_Constellation.reshape(-1) * self.dqam
        self.constellation_norm = tf.convert_to_tensor(self.constellation_norm)
        self.loss = trainset.loss
        self.iter = trainset.cg_iter
        # trainable variables
        self.alpha, self.beta = [], []
        for t in range(self.iter):
            self.alpha.append(tf.Variable(0.0, dtype=tf.float64, name='alpha_' + str(t), trainable=True))
            if t != self.iter - 1:
                self.beta.append(tf.Variable(0.0, dtype=tf.float64, name='beta_' + str(t), trainable=True))

    def __call__(self, x, y, A, noise_var, bs=None, test=False):
        xhat = tf.zeros_like(x, tf.complex128)
        noise_var = tf.cast(noise_var, dtype=tf.complex128)
        AH = tf.linalg.adjoint(A)
        AHA = tf.matmul(AH, A)
        ytilde = tf.matmul(AH, y)
        xi = AHA + noise_var * tf.eye(self.nt, batch_shape=[bs], dtype=tf.complex128)
        r = ytilde - tf.matmul(xi, xhat)  # (bs, nt, 1)
        di = r
        # r_norm = tf.norm(r, axis=1, keepdims=True)  # (bs, 1, 1)

        loss = 0.
        mse = []
        for i in range(self.iter):
            # compute the approximate solution based on prior conjugate direction and residual
            xi_di = tf.matmul(xi, di)
            # alpha = r_norm ** 2 / tf.matmul(tf.linalg.adjoint(di), xi_di)
            xhat = xhat + tf.cast(self.alpha[i], dtype=tf.complex128) * di

            mse.append((tf.nn.l2_loss(tf.math.real(xhat) - tf.math.real(x)) +
                      tf.nn.l2_loss(tf.math.imag(xhat) - tf.math.imag(x)))
                       / tf.cast(bs * self.nt / 2, dtype=tf.float64))
            if i == self.iter - 1:
                continue
            # compute conjugate direction and residual
            r -= tf.cast(self.alpha[i], dtype=tf.complex128) * xi_di
            # r_norm_last = r_norm
            # r_norm = tf.norm(r, axis=1, keepdims=True)
            # beta = r_norm ** 2 / r_norm_last ** 2
            di = r + tf.cast(self.beta[i], dtype=tf.complex128) * di

        if self.loss == 'mse':
            loss += mse[self.iter - 1]

        return xhat, loss, tf.concat([mse], axis=0)


