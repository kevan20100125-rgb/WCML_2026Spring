#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train_tf2 import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
import numpy as np
import numpy.linalg as la
import sys
import time
import tensorflow as tf
import scipy.io as sio

from tensorflow.keras import Model


def train_mhcg_net(test=False, trainset = None):
    Mr, Nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size
    savefile = trainset.savefile
    directory = './model/' + 'MHCG_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    model = MHCGNet(trainset=trainset)

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


class MHCGNet(Model):
    def __init__(self, trainset=None):
        super(MHCGNet, self).__init__()
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
        self.iter, self.samplers = trainset.samples, trainset.samplers
        self.mmse_init = trainset.mmse_init
        self.d_tune, self.vec_step_size = trainset.d_tune, trainset.vec_step_size
        # trainable variables
        self.gamma = []
        for t in range(self.iter):
            if self.d_tune:
                # self.gamma.append(tf.Variable(1.0, dtype=tf.float64, name='gamma_' + str(t), trainable=True))
                self.gamma.append(tf.Variable(- tf.math.log(self.es - 1), dtype=tf.float64,
                                              name='gamma_' + str(t), trainable=True))

    def __call__(self, x, y, A, noise_var, bs=None, test=False):
        AH = tf.linalg.adjoint(A)
        AHA = tf.matmul(AH, A)
        noise_var = tf.cast(noise_var, dtype=tf.complex128)
        xi = AHA + noise_var * tf.eye(self.nt, batch_shape=[bs], dtype=tf.complex128)
        ytilde = tf.matmul(AH, y)
        eta = 1
        # if self.mr != self.nt:
        #     Ainv = tf.linalg.cholesky(tf.linalg.inv(AHA))
        # else:
        #     Ainv = tf.linalg.inv(A)
        # col_norm = 1 / tf.norm(Ainv, axis=1, keepdims=True)  # (bs, 1, nt) complex128
        # covar = Ainv * col_norm
        covar = tf.eye(self.nt, dtype=tf.complex128)

        if self.mmse_init is True:
            x_mmse = tf.matmul(tf.matmul(tf.linalg.inv(AHA + noise_var *
                                                       tf.eye(self.nt, batch_shape=[bs], dtype=tf.complex128)), AH), y)
            # quantization
            indices = tf.argmin(abs(x_mmse * np.ones((1, self.nt, 2 ** self.mu))
                                    - self.constellation_norm), axis=2)  # (bs, nt)
            xhat = tf.gather_nd(self.constellation_norm, indices[:, :, tf.newaxis])[:, :, tf.newaxis]  # (bs, nt, 1)
            if test:
                # xmmse = tf.reshape(tf.tile(xhat, [self.samplers, 1, 1]), [self.samplers, bs, self.nt, 1])
                # indices = tf.random.uniform(shape=[self.samplers, bs, self.nt, 1], maxval=2 ** self.mu, dtype=tf.int64)
                # xhat = tf.gather_nd(self.constellation_norm, indices)[:, :, :, tf.newaxis]  # (np, bs, nt, 1)
                # mask = tf.reshape(tf.convert_to_tensor([True, False, False, False, False, False, False, False,
                #                                         False, False, False, False, False, False, False, False]),
                #                   [16, 1, 1, 1])
                # mask = tf.tile(mask, [1, bs, self.nt, 1])
                # xhat = tf.where(condition=mask, x=xmmse, y=xhat)  # only one mmse
                xhat = tf.reshape(tf.tile(xhat, [self.samplers, 1, 1]), [self.samplers, bs, self.nt, 1])  # all mmse
        else:
            indices = tf.random.uniform(shape=[self.samplers, bs, self.nt, 1], maxval=2 ** self.mu, dtype=tf.int64)
            xhat = tf.gather_nd(self.constellation_norm, indices)[:, :, :, tf.newaxis]  # (np, bs, nt, 1)


        r = y - tf.matmul(A, xhat)
        r_norm = tf.cast(tf.norm(r, axis=2, keepdims=True) ** 2, dtype=tf.float64)  # (np, bs, 1, 1)
        x_survivor = xhat
        r_norm_survivor = r_norm

        xhat_cg = xhat
        r_cg = ytilde - tf.matmul(xi, xhat_cg)  # (bs, nt, 1)
        r_norm_cg = tf.cast(tf.norm(r_cg, axis=2, keepdims=True) ** 2, dtype=tf.float64)  # (np, bs, 1, 1)
        di = r_cg

        dqam_for_step = self.dqam

        # core
        loss = 0.
        mse = []
        for t in range(self.iter):
            # compute the approximate solution based on prior conjugate direction and residual
            xi_di = tf.matmul(xi, di)
            alpha = r_norm_cg / tf.cast(tf.matmul(tf.linalg.adjoint(di), xi_di), dtype=tf.float64)
            xhat_cg = xhat_cg + tf.cast(alpha, dtype=tf.complex128) * di

            if self.d_tune:
                # dqam_for_step *= tf.minimum(self.gamma[0], self.es)  # dqam for step less than one to avoid too large jump
                dqam_for_step = self.dqam * (self.es * tf.math.sigmoid(self.gamma[t]))
            if self.vec_step_size:
                step_size = tf.maximum(dqam_for_step, abs(tf.matmul(AH, r))
                                       / tf.math.sqrt(tf.cast(self.nt, dtype=tf.float64))) * eta  # (bs, nt, 1) float64
            else:
                step_size = tf.maximum(dqam_for_step,
                                       tf.math.sqrt(r_norm) / self.nt) * eta  # (bs, 1, 1) float64
            # construct the proposal
            v = tf.dtypes.complex(tf.random.normal([self.samplers, bs, self.nt, 1], dtype=tf.float64),
                                  tf.random.normal([self.samplers, bs, self.nt, 1], dtype=tf.float64)) \
                / np.sqrt(2)  # zero-mean, unit variance
            # construct the proposal
            z_prop = xhat_cg + tf.matmul(tf.cast(step_size, dtype=tf.complex128) * covar, v)

            # quantization
            if test:  # ideal quantization for testing
                indices = tf.argmin(abs(z_prop * np.ones((1, 1, self.nt, 2 ** self.mu))
                                        - self.constellation_norm), axis=3)  # (np, bs, nt)
                x_prop = tf.gather_nd(self.constellation_norm, indices[:, :, :, tf.newaxis])[:, :, :, tf.newaxis]  # (np, bs, nt, 1)
            else:   # approximate quantization for training
                real, imag = tf.math.real(z_prop), tf.math.imag(z_prop)
                x_real, x_imag = self.approx_quantization(real), self.approx_quantization(imag)
                x_prop = tf.dtypes.complex(x_real, x_imag)
            r_prop = y - tf.matmul(A, x_prop)  # todo: use the calculation for CG iterations
            r_norm_prop = tf.cast(tf.norm(r_prop, axis=2, keepdims=True) ** 2, dtype=tf.float64)  # (np, bs, 1, 1)

            # update survivor
            update = r_norm_prop < r_norm_survivor  # (np, bs, 1, 1)
            x_survivor = tf.where(condition=tf.tile(update, [1, 1, self.nt, 1]), x=x_prop, y=x_survivor)
            r_norm_survivor = tf.where(condition=update, x=r_norm_prop, y=r_norm_survivor)

            # acceptance step
            log_pacc = tf.minimum(tf.cast(0., dtype=tf.float64),
                                  -(r_norm_prop - r_norm) / tf.cast((2), dtype=tf.float64))  # float64
            p_acc = tf.exp(log_pacc)  # float64
            p_uni = tf.random.uniform([self.samplers, bs, 1, 1], maxval=1.0, dtype=tf.float64)  # float64
            mask = (p_acc >= p_uni)  # (bs, 1 ,1)
            xhat = tf.where(condition=tf.tile(mask, [1, 1, self.nt, 1]), x=x_prop, y=xhat)
            r = tf.where(condition=tf.tile(mask, [1, 1, self.nt, 1]), x=r_prop, y=r)
            r_norm = tf.where(condition=mask, x=r_norm_prop, y=r_norm)

            # compute conjugate direction and residual
            r_cg -= tf.cast(alpha, dtype=tf.complex128) * xi_di
            r_norm_cg_last = r_norm_cg
            r_norm_cg = tf.cast(tf.norm(r_cg, axis=2, keepdims=True) ** 2, dtype=tf.float64)  # (np, bs, 1, 1)
            beta = r_norm_cg / r_norm_cg_last  # (np, 1, 1)
            di = r_cg + tf.cast(beta, dtype=tf.complex128) * di  # (bs, nt, 1)

            mse.append((tf.nn.l2_loss(tf.math.real(xhat) - tf.math.real(x)) +
                        tf.nn.l2_loss(tf.math.imag(xhat) - tf.math.imag(x)))
                        / tf.cast(bs * self.samplers * self.nt / 2, dtype=tf.float64))

            if ((t + 1) % self.nt) == 0:  # restart CG
                xhat_cg = x_survivor
                r_cg = ytilde - tf.matmul(xi, xhat_cg)
                r_norm_cg = tf.cast(tf.norm(r_cg, axis=2, keepdims=True) ** 2, dtype=tf.float64)  # (np, bs, 1, 1)
                di = r_cg
            # if self.loss == 'mse':
            #     loss += mse[t]
            # elif self.loss == 'norm':
            #     loss += tf.reduce_sum(tf.math.sqrt(tf.cast(r_norm, dtype=tf.float64))) / tf.cast(bs, dtype=tf.float64)

        # select the sample that minimizes the ML cost
        # idx = tf.cast(tf.argmin(tf.cast(r_norm, dtype=tf.float64), axis=0), dtype=tf.int32)
        # xhat = tf2.experimental.numpy.take_along_axis(xhat, idx[tf.newaxis, :], axis=0)
        # r_norm_survivor = tf.norm(y - tf.matmul(A, x_survivor), axis=2, keepdims=True) ** 2
        idx = tf.cast(tf.argmin(tf.cast(r_norm_survivor, dtype=tf.float64), axis=0), dtype=tf.int32)
        xhat = tf.experimental.numpy.take_along_axis(x_survivor, idx[tf.newaxis, :], axis=0)
        xhat = tf.squeeze(xhat, axis=0)
        # xhat = xhat[tf.squeeze(tf.argmin(tf.cast(r_norm, dtype=tf.float64), axis=0)), :, :, :]
        # todo: average loss
        if self.loss == 'mse':
            loss += mse[self.iter - 1]
        elif self.loss == 'norm':
            r_norm_survivor = tf.reduce_min(r_norm_survivor, axis=0)
            # r_norm_survivor = (tf.reduce_sum(r_norm_survivor, axis=0) + r_norm_min * 1000) / (1000 + self.samplers - 1)
            loss += tf.reduce_sum(tf.math.sqrt(tf.cast(r_norm_survivor, dtype=tf.float64))) / tf.cast(bs, dtype=tf.float64)
        return xhat, loss, tf.concat([mse], axis=0)

    def approx_quantization(self, x):
        m = int(np.sqrt(2 ** self.mu))
        eta = tf.constant(100.0, dtype=tf.float64)
        y = - (m - 1)
        for i in range(1, m):
            y += tf.math.tanh(eta * (x - (2 * i - m) / self.es)) + 1
        y /= self.es
        return y