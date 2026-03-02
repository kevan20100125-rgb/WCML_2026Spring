#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
import numpy as np
import numpy.linalg as la
import sys
import time
import tensorflow.compat.v1 as tf
import scipy.io as sio

tf.disable_v2_behavior()


def train_mhngd_net(test=False, trainset = None):
    Mr, Nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size
    savefile = trainset.savefile
    directory = './model/' + 'MHNGD_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainset.prob
    x_, y_, H_, sigma2_, bs_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.sample_size_
    model = MHNGDNet(trainset=trainset)
    xhat, loss_, mse = model.build(x_, y_, H_, sigma2_, bs=bs_, test=test)  # transfer place holder and build the model

    train, grads_ = [], []
    global_step = tf.Variable(0, trainable=False)
    lr_ = tf.train.exponential_decay(lr, global_step, decay_steps, lr_decay, name='lr')

    if test is False:
        grads_, _ = tf.clip_by_global_norm(tf.gradients(loss_, tf.trainable_variables()),
                                           trainset.grad_clip)
        if trainset.grad_clip_flag:
            optimizer = tf.train.AdamOptimizer(lr_)
            # grads_, _ = tf.clip_by_global_norm(tf.gradients(loss_, tf.trainable_variables()),
            #                                    trainset.grad_clip)
            if tf.trainable_variables():
                train = optimizer.apply_gradients(zip(grads_, tf.trainable_variables()), global_step)
        else:
            if tf.trainable_variables():
                train = tf.train.AdamOptimizer(lr_).minimize(loss_, global_step, var_list=tf.trainable_variables())
                # train = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)

    if test:
        return sess, xhat, mse

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
                loss_batch = sess.run(loss_, feed_dict={y_: ybatch,
                                                        x_: xbatch, H_: Hbatch, sigma2_: sigma2batch,
                                                        bs_: val_batch_size})
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


class MHNGDNet:
    def __init__(self, trainset=None):
        self.mr, self.nt, self.mu = trainset.m, trainset.n, trainset.mu
        self.dqam = np.sqrt(3 / 2 / (2 ** self.mu - 1))
        self.noise_scale = (2 * self.dqam) ** 2 / np.log(2)
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
        self.lr_approx, self.mmse_init = trainset.lr_approx, trainset.mmse_init
        self.d_tune, self.vec_step_size = trainset.d_tune, trainset.vec_step_size
        self.nolr = trainset.nolr
        self.cg_iter = trainset.cg_iter
        self.gd, self.learnable_beta = trainset.gd, trainset.beta
        self.quantize, self.post = trainset.quantize, trainset.post
        self.early_stop = trainset.es
        # trainable variables
        self.lr, self.gamma, self.beta = [], [], []
        for t in range(self.iter):
            if self.nolr is True:
                self.lr.append(0)
            else:
                self.lr.append(tf.Variable(0.25, dtype=tf.float64, name='lr_' + str(t), trainable=True))
            if self.d_tune:
                # self.gamma.append(tf.Variable(1.0, dtype=tf.float64, name='gamma_' + str(t), trainable=True))
                self.gamma.append(tf.Variable(- tf.log(self.es - 1), dtype=tf.float64,
                                              name='gamma_' + str(t), trainable=True))
            if self.learnable_beta:
                for i in range(self.gd):
                    self.beta.append(tf.Variable(tf.cast(- tf.log(1 / 0.9 - 1), dtype=tf.float64), dtype=tf.float64,
                                                 name='beta_' + str(t * self.gd + i), trainable=True))

    def build(self, x, y, A, noise_var, bs=None, test=False, x_fs=None):
        AH = tf.linalg.adjoint(A)
        AHA = tf.matmul(AH, A)
        noise_var = tf.cast(noise_var, dtype=tf.complex128)
        alpha = 1
        if self.mr != self.nt:
            Ainv = tf.linalg.cholesky(tf.linalg.inv(AHA))
        else:
            Ainv = tf.linalg.inv(A)
        col_norm = 1 / tf.norm(Ainv, axis=1, keepdims=True)  # (bs, 1, nt) complex128
        covar = Ainv * col_norm
        # covar = tf.eye(self.nt, dtype=tf.complex128)  # todo

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
                # xhat = tf.where(condition=mask, x=xmmse, y=xhat)
                xhat = tf.reshape(tf.tile(xhat, [self.samplers, 1, 1]), [self.samplers, bs, self.nt, 1])
        elif self.mmse_init == 'fs':
            # xhat = x_fs  # (bs, nt, 1)
            # # quantization
            indices = tf.argmin(abs(x_fs * np.ones((1, self.nt, 2 ** self.mu))
                                    - self.constellation_norm), axis=2)  # (bs, nt)
            xhat = tf.gather_nd(self.constellation_norm, indices[:, :, tf.newaxis])[:, :, tf.newaxis]  # (bs, nt, 1)
            if test:
                # x_fs = tf.reshape(tf.tile(xhat, [self.samplers, 1, 1]), [self.samplers, bs, self.nt, 1])
                # indices = tf.random.uniform(shape=[self.samplers, bs, self.nt, 1], maxval=2 ** self.mu, dtype=tf.int64)
                # xhat = tf.gather_nd(self.constellation_norm, indices)[:, :, :, tf.newaxis]  # (np, bs, nt, 1)
                # mask = np.zeros(self.samplers, dtype=bool)
                # mask[0] = True
                # mask = tf.reshape(tf.convert_to_tensor(mask), [self.samplers, 1, 1, 1])
                # mask = tf.tile(mask, [1, bs, self.nt, 1])
                # xhat = tf.where(condition=mask, x=x_fs, y=xhat)
                xhat = tf.reshape(tf.tile(xhat, [self.samplers, 1, 1]), [self.samplers, bs, self.nt, 1])
        elif self.mmse_init == 'cg':
            xinit = tf.zeros_like(x, dtype=tf.complex128)
            x_mmse = self.cg(xinit, xi=AHA + noise_var *
                                       tf.eye(self.nt, batch_shape=[bs], dtype=tf.complex128),
                             y=tf.matmul(AH, y), iter=self.cg_iter)
            # quantization
            indices = tf.argmin(abs(x_mmse * np.ones((1, self.nt, 2 ** self.mu))
                                    - self.constellation_norm), axis=2)  # (bs, nt)
            xhat = tf.gather_nd(self.constellation_norm, indices[:, :, tf.newaxis])[:, :, tf.newaxis]  # (bs, nt, 1)
            if test:
                xhat = tf.reshape(tf.tile(xhat, [self.samplers, 1, 1]), [self.samplers, bs, self.nt, 1])
        else:
            indices = tf.random.uniform(shape=[self.samplers, bs, self.nt, 1], maxval=2 ** self.mu, dtype=tf.int64)
            xhat = tf.gather_nd(self.constellation_norm, indices)[:, :, :, tf.newaxis]  # (np, bs, nt, 1)

        r = y - tf.matmul(A, xhat)  # (bs, nr, 1)
        r_norm = tf.cast(tf.norm(r, axis=2, keepdims=True) ** 2, dtype=tf.float64)  # (np, bs, 1, 1)
        x_survivor = xhat
        r_norm_survivor = r_norm

        dqam_for_step = self.dqam
        if self.d_tune:
            # dqam_for_step *= tf.minimum(self.gamma[0], self.es)  # dqam for step less than one to avoid too large jump
            dqam_for_step *= (
                    self.es * tf.math.sigmoid(self.gamma[0]))  # dqam for step less than one to avoid too large jump
        if self.vec_step_size:
            step_size = tf.maximum(dqam_for_step, abs(tf.matmul(AH, r))
                                   / tf.math.sqrt(tf.cast(self.nt, dtype=tf.float64))) * alpha  # (bs, nt, 1) float64
        else:
            step_size = tf.maximum(dqam_for_step,
                                   tf.math.sqrt(r_norm) / self.nt) * alpha  # (bs, 1, 1) float64

        if self.nolr:
            L = tf.norm(AHA, axis=[-2, -1], ord='fro', keepdims=True)
            self.lr = 1 / L
        # core
        loss = 0.
        mse = []
        momentum = 0.
        beta = 0.9
        for t in range(self.iter):
            z_grad = xhat
            # construct the proposal
            for i in range(self.gd):
                z_grad_last = z_grad
                if self.learnable_beta:
                    y_grad = z_grad + tf.cast(tf.math.sigmoid(self.beta[t * self.gd + i]),
                                              dtype=tf.complex128) * momentum
                else:
                    y_grad = z_grad + beta * momentum
                if self.nolr:
                    z_grad = y_grad + tf.matmul(tf.cast(self.lr, dtype=tf.complex128) * AH, y - tf.matmul(A, y_grad))
                else:
                    z_grad = y_grad + tf.matmul(tf.cast(self.lr[t], dtype=tf.complex128) * AH, y - tf.matmul(A, y_grad))
                momentum = z_grad - z_grad_last
                if test and self.quantize and i != self.gd - 1 and t >= self.post:
                    indices = tf.argmin(abs(z_grad * np.ones((1, 1, self.nt, 2 ** self.mu))
                                            - self.constellation_norm), axis=3)  # (np, bs, nt)
                    x_prop = tf.gather_nd(self.constellation_norm, indices[:, :, :, tf.newaxis])[:, :, :,
                             tf.newaxis]  # (np, bs, nt, 1)
                    r_prop = y - tf.matmul(A, x_prop)
                    r_norm_prop = tf.cast(tf.norm(r_prop, axis=2, keepdims=True) ** 2,
                                              dtype=tf.float64)  # (np, bs, 1, 1))
                    update = r_norm_prop < r_norm_survivor  # (bs, 1 ,1) or (np, bs, 1, 1)
                    x_survivor = tf.where(condition=tf.tile(update, [1, 1, self.nt, 1]), x=x_prop, y=x_survivor)
                    r_norm_survivor = tf.where(condition=update, x=r_norm_prop, y=r_norm_survivor)

            v = tf.dtypes.complex(tf.random.normal([self.samplers, bs, self.nt, 1], dtype=tf.float64),
                                  tf.random.normal([self.samplers, bs, self.nt, 1], dtype=tf.float64)) \
                / np.sqrt(2)  # zero-mean, unit variance
            z_prop = z_grad + tf.matmul(tf.cast(step_size, dtype=tf.complex128) * covar, v)
            # quantization
            if test:  # ideal quantization for testing
                indices = tf.argmin(abs(z_prop * np.ones((1, 1, self.nt, 2 ** self.mu))
                                        - self.constellation_norm), axis=3)  # (np, bs, nt)
                x_prop = tf.gather_nd(self.constellation_norm, indices[:, :, :, tf.newaxis])[:, :, :,
                         tf.newaxis]  # (np, bs, nt, 1)
            else:  # approximate quantization for training
                real, imag = tf.math.real(z_prop), tf.math.imag(z_prop)
                x_real, x_imag = self.approx_quantization(real), self.approx_quantization(imag)
                x_prop = tf.dtypes.complex(x_real, x_imag)
            r_prop = y - tf.matmul(A, x_prop)
            r_norm_prop = tf.cast(tf.norm(r_prop, axis=2, keepdims=True) ** 2, dtype=tf.float64)  # (np, bs, 1, 1)

            update = r_norm_prop < r_norm_survivor  # (bs, 1 ,1) or (np, bs, 1, 1)
            x_survivor = tf.where(condition=tf.tile(update, [1, 1, self.nt, 1]), x=x_prop, y=x_survivor)
            r_norm_survivor = tf.where(condition=update, x=r_norm_prop, y=r_norm_survivor)

            # acceptance step
            log_pacc = tf.minimum(tf.cast(0., dtype=tf.float64),
                                  -(r_norm_prop - r_norm) / tf.cast((1), dtype=tf.float64))  # float64
            p_acc = tf.exp(log_pacc)  # float64
            p_uni = tf.random.uniform([self.samplers, bs, 1, 1], maxval=1.0, dtype=tf.float64)  # float64
            mask = (p_acc >= p_uni)  # (bs, 1 ,1) or (np, bs, 1, 1)
            xhat = tf.where(condition=tf.tile(mask, [1, 1, self.nt, 1]), x=x_prop, y=xhat)
            r = tf.where(condition=tf.tile(mask, [1, 1, self.mr, 1]), x=r_prop, y=r)
            r_norm = tf.where(condition=mask, x=r_norm_prop, y=r_norm)
            mse.append((tf.nn.l2_loss(tf.math.real(xhat) - tf.math.real(x)) +
                        tf.nn.l2_loss(tf.math.imag(xhat) - tf.math.imag(x)))
                       / tf.cast(bs * self.samplers / 2, dtype=tf.float64))
            # if self.loss == 'mse':
            #     loss += mse[t]
            # elif self.loss == 'norm':
            #     loss += tf.reduce_sum(tf.math.sqrt(tf.cast(r_norm, dtype=tf.float64))) / tf.cast(bs, dtype=tf.float64)

            if t == self.iter - 1:  # skip the final iteration for ineffective calculation of step size
                continue

            if self.d_tune:
                # dqam_for_step = self.dqam * tf.minimum(self.gamma[t + 1], self.es)
                dqam_for_step = self.dqam * (self.es * tf.math.sigmoid(
                    self.gamma[t + 1]))  # dqam for step less than one to avoid too large jump
            if self.vec_step_size:
                step_size = tf.where(condition=tf.tile(mask, [1, 1, self.nt, 1]),
                                     x=tf.maximum(abs(tf.matmul(AH, r))
                                                  / tf.math.sqrt(tf.cast(self.nt, dtype=tf.float64)),
                                                  dqam_for_step) * alpha,
                                     y=step_size)  # (np, bs, nt, 1)
            else:
                step_size = tf.where(condition=mask,
                                     x=tf.maximum(tf.math.sqrt(r_norm) / self.nt, dqam_for_step) * alpha,
                                     y=step_size)

        # select the sample that minimizes the ML cost
        if test:
            import tensorflow as tf2
            # idx = tf.cast(tf.argmin(tf.cast(r_norm, dtype=tf.float64), axis=0), dtype=tf.int32)
            # xhat = tf2.experimental.numpy.take_along_axis(xhat, idx[tf.newaxis, :], axis=0)
            # r_norm_survivor = tf.norm(y - tf.matmul(A, x_survivor), axis=2, keepdims=True) ** 2
            idx = tf.cast(tf.argmin(tf.cast(r_norm_survivor, dtype=tf.float64), axis=0), dtype=tf.int32)
            xhat = tf2.experimental.numpy.take_along_axis(x_survivor, idx[tf.newaxis, :], axis=0)
            xhat = tf.squeeze(xhat, axis=0)
            # xhat = xhat[tf.squeeze(tf.argmin(tf.cast(r_norm, dtype=tf.float64), axis=0)), :, :, :]
        # todo: final loss
        if self.loss == 'mse':
            loss += mse[self.iter - 1]
        elif self.loss == 'norm':
            # optimize for parallel samplers
            r_norm_survivor = tf.reduce_min(r_norm_survivor, axis=0)  # (bs, 1, 1)
            loss += tf.reduce_sum(tf.math.sqrt(r_norm_survivor)) / tf.cast(bs, dtype=tf.float64)
        return xhat, loss, tf.concat([mse], axis=0)

    def cg(self, xinit, xi, y, iter=8):
        r = y - tf.matmul(xi, xinit)
        xhat = xinit
        di = r
        r_norm = tf.norm(r, axis=1, keepdims=True)  # (bs, 1, 1)

        for t in range(iter):
            # compute the approximate solution based on prior conjugate direction and residual
            xi_di = tf.matmul(xi, di)
            alpha = r_norm ** 2 / tf.matmul(tf.linalg.adjoint(di), xi_di)
            xhat += alpha * di

            # compute conjugate direction and residual
            r -= alpha * xi_di
            r_norm_last = r_norm
            r_norm = tf.norm(r, axis=1, keepdims=True)
            beta = r_norm ** 2 / r_norm_last ** 2
            di = r + beta * di

        return xhat

    def approx_quantization(self, x):
        m = int(np.sqrt(2 ** self.mu))
        eta = tf.constant(100.0, dtype=tf.float64)
        y = - (m - 1)
        for i in range(1, m):
            y += tf.math.tanh(eta * (x - (2 * i - m) / self.es)) + 1
        y /= self.es
        return y