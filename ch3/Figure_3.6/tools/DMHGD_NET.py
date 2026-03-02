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


def train_dmhgd_net(test=False, trainset = None):
    Mr, Nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size
    savefile = trainset.savefile
    directory = './model/' + 'DMHGD_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainset.prob
    x_, y_, H_, sigma2_, bs_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.sample_size_
    model = DMHGDNet(trainset=trainset)
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


class DMHGDNet:
    def __init__(self, trainset=None):
        self.mr, self.nt, self.mu, self.c = trainset.m, trainset.n, trainset.mu, trainset.c
        self.n = self.mr // self.c
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
        self.mmse_init = trainset.mmse_init
        self.d_tune, self.vec_step_size = trainset.d_tune, trainset.vec_step_size
        self.cg_iter = trainset.cg_iter
        # trainable variables
        self.lr, self.gamma = [], []
        self.sigma = []
        for t in range(self.iter):
            self.lr.append(tf.Variable(1.0, dtype=tf.float64, name='lr_' + str(t), trainable=True))
            if self.d_tune:
                # self.gamma.append(tf.Variable(1.0, dtype=tf.float64, name='gamma_' + str(t), trainable=True))
                self.gamma.append(tf.Variable(- tf.log(self.es - 1), dtype=tf.float64,
                                              name='gamma_' + str(t), trainable=True))

    def build(self, x, y, A, noise_var, bs=None, test=False):
        yc = tf.reshape(y, [bs, self.c, self.n, 1])
        Ac = tf.reshape(A, [bs, self.c, self.n, self.nt])
        AcH = tf.linalg.adjoint(Ac)  # (bs, c, nt, n)
        AcHAc = tf.matmul(AcH, Ac)  # (bs, c, nt, nt)
        noise_var = tf.cast(noise_var, dtype=tf.complex128)
        AHA = tf.reduce_sum(AcHAc, axis=1)  # (bs, nt, nt)
        grad_preconditioner = tf.linalg.inv(AHA + noise_var / (self.dqam ** 2) *
                                            tf.eye(self.nt, batch_shape=[bs], dtype=tf.complex128))  # (bs, nt, nt)
        alpha = 0.5
        if self.mr != self.nt:
            Ainv = tf.linalg.cholesky(tf.linalg.inv(AHA))
        else:
            Ainv = tf.linalg.inv(A)
        col_norm = 1 / tf.norm(Ainv, axis=1, keepdims=True)  # (bs, 1, nt) complex128
        covar = Ainv * col_norm  # (bs, nt, nt)
        # covar = tf.eye(self.nt, dtype=tf.complex128)

        if self.mmse_init is True:
            AH = tf.linalg.adjoint(A)
            x_mmse = tf.matmul(tf.matmul(tf.linalg.inv(AHA + noise_var *
                                                       tf.eye(self.nt, batch_shape=[bs], dtype=tf.complex128)), AH), y)
            # quantization
            indices = tf.argmin(abs(x_mmse * np.ones((1, self.nt, 2 ** self.mu))
                                                                    - self.constellation_norm), axis=2)  # (bs, nt)
            xhat = tf.gather_nd(self.constellation_norm, indices[:, :, tf.newaxis])[:, :, tf.newaxis]  # (bs, nt, 1)
            if test:
                xhat = tf.reshape(tf.tile(xhat, [self.samplers, 1, 1]), [self.samplers, bs, self.nt, 1])  # all mmse
        else:
            indices = tf.random.uniform(shape=[bs, self.nt, self.samplers, 1], maxval=2 ** self.mu, dtype=tf.int64)
            xhat = tf.gather_nd(self.constellation_norm, indices)  # (bs, nt, np)

        r = y - tf.matmul(A, xhat)  # (bs, nr, np)
        rc = yc - tf.einsum('ijkl, ilm->ijkm', Ac, xhat)  # (bs, c, n, np)
        r_norm = tf.reduce_sum(tf.abs(r) ** 2, axis=1)  # (bs, np)
        x_survivor, r_norm_survivor = xhat, r_norm

        dqam_for_step = self.dqam
        if self.d_tune:
            # dqam_for_step *= tf.minimum(self.gamma[0], self.es)  # dqam for step less than one to avoid too large jump
            alpha = 0.5 * (self.es * tf.math.sigmoid(self.gamma[0]))  # dqam for step less than one to avoid too large jump
        if self.vec_step_size:
            AH = tf.linalg.adjoint(A)
            step_size = tf.maximum(dqam_for_step, abs(tf.matmul(AH, r))
                                   / tf.math.sqrt(tf.cast(self.nt, dtype=tf.float64))) * alpha  # (bs, nt, 1) float64
        else:
            step_size = tf.maximum(dqam_for_step,
                                   tf.math.sqrt(r_norm) / self.nt) * alpha  # (bs, np) float64

        # core
        loss = 0.
        mse = []
        for t in range(self.iter):
            # construct the proposal
            grad = tf.reduce_sum(tf.matmul(AcH, rc), axis=1)  # (bs, nt, np)
            z_grad = xhat + tf.cast(tf.minimum(self.lr[t], 5), dtype=tf.complex128) * \
                     tf.matmul(grad_preconditioner, grad) # constrain lr to be too large, (bs, nt, np)

            v = tf.dtypes.complex(tf.random.normal([bs, self.nt, self.samplers], dtype=tf.float64),
                                  tf.random.normal([bs, self.nt, self.samplers], dtype=tf.float64)) / np.sqrt(2)  # zero-mean, unit variance
            z_prop = z_grad + tf.einsum('ik, ijk->ijk', tf.cast(step_size, dtype=tf.complex128), tf.matmul(covar, v))  # (bs, nt, np)
            # quantization
            if test:  # ideal quantization for testing
                indices = tf.argmin(abs(z_prop[:, :, :, tf.newaxis] * np.ones((2 ** self.mu, ))
                                        - self.constellation_norm), axis=3)  # (bs, nt, np)
                x_prop = tf.gather_nd(self.constellation_norm, indices[:, :, :, tf.newaxis])  # (bs, nt, np)
            else:   # approximate quantization for training
                real, imag = tf.math.real(z_prop), tf.math.imag(z_prop)
                x_real, x_imag = self.approx_quantization(real), self.approx_quantization(imag)
                x_prop = tf.dtypes.complex(x_real, x_imag)
            r_prop = y - tf.matmul(A, x_prop)  # (bs, nr, np)
            r_norm_prop = tf.reduce_sum(tf.abs(r_prop) ** 2, axis=1)  # (bs, np)

            update = r_norm_prop < r_norm_survivor  # (bs, np)
            x_survivor = tf.where(condition=tf.tile(update[:, tf.newaxis, :], [1, self.nt, 1]), x=x_prop, y=x_survivor)
            r_norm_survivor = tf.where(condition=update, x=r_norm_prop, y=r_norm_survivor)

            # acceptance step
            log_pacc = tf.minimum(tf.cast(0., dtype=tf.float64),
                                  -(r_norm_prop - r_norm) / tf.cast((1), dtype=tf.float64))  # float64  (bs, np)
            p_acc = tf.exp(log_pacc)
            p_uni = tf.random.uniform([bs, self.samplers], maxval=1.0, dtype=tf.float64)  # float64
            mask = (p_acc >= p_uni)  # (bs, np)
            xhat = tf.where(condition=tf.tile(mask[:, tf.newaxis, :], [1, self.nt, 1]), x=x_prop, y=xhat)
            r = tf.where(condition=tf.tile(mask[:, tf.newaxis, :], [1, self.mr, 1]), x=r_prop, y=r)
            rc = yc - tf.einsum('ijkl, ilm->ijkm', Ac, xhat)  # update rc !
            r_norm = tf.where(condition=mask, x=r_norm_prop, y=r_norm)

            mse.append((tf.nn.l2_loss(tf.math.real(x_survivor) - tf.math.real(x)) +
                        tf.nn.l2_loss(tf.math.imag(x_survivor) - tf.math.imag(x)))
                       / tf.cast(bs * self.samplers * self.nt / 2, dtype=tf.float64))

            if t == self.iter - 1:  # skip the final iteration for ineffective calculation of step size
                continue
            if self.d_tune:
                # dqam_for_step = self.dqam * tf.minimum(self.gamma[t + 1], self.es)
                alpha = 0.5 * (self.es * tf.math.sigmoid(
                    self.gamma[t + 1]))  # dqam for step less than one to avoid too large jump
            if self.vec_step_size:
                AH = tf.linalg.adjoint(A)
                step_size = tf.where(condition=tf.tile(mask, [1, 1, self.nt, 1]),
                                     x=tf.maximum(abs(tf.matmul(AH, r))
                                                  / tf.math.sqrt(tf.cast(self.nt, dtype=tf.float64)), dqam_for_step) * alpha,
                                     y=step_size) # (np, bs, nt, 1)
            else:
                step_size = tf.where(condition=mask,
                                     x=tf.maximum(tf.math.sqrt(r_norm) / self.nt, dqam_for_step) * alpha,
                                     y=step_size)  # (bs, np) float64

        # select the sample that minimizes the ML cost
        import tensorflow as tf2
        # idx = tf.cast(tf.argmin(tf.cast(r_norm, dtype=tf.float64), axis=0), dtype=tf.int32)
        # xhat = tf2.experimental.numpy.take_along_axis(xhat, idx[tf.newaxis, :], axis=0)
        # r_norm_survivor = tf.norm(y - tf.matmul(A, x_survivor), axis=2, keepdims=True) ** 2
        idx = tf.cast(tf.argmin(tf.cast(r_norm_survivor, dtype=tf.float64), axis=1), dtype=tf.int32)  # (bs, )
        xhat = tf2.experimental.numpy.take_along_axis(x_survivor, idx[:, tf.newaxis, tf.newaxis], axis=2)  # (bs, nt, 1)
        # xhat = tf.squeeze(xhat, axis=0)

        if self.loss == 'mse':
            loss += mse[self.iter - 1]
        elif self.loss == 'norm':
            r_norm_survivor = tf.reduce_min(r_norm_survivor, axis=1)  # (bs, )
            # r_norm_survivor = (tf.reduce_sum(r_norm_survivor, axis=0) + r_norm_min * 1000) / (1000 + self.samplers - 1)
            loss += tf.reduce_sum(tf.math.sqrt(tf.cast(r_norm_survivor, dtype=tf.float64))) / tf.cast(bs,
                                                                                                      dtype=tf.float64)
        return xhat, loss, tf.concat([mse], axis=0)

    def approx_quantization(self, x):
        m = int(np.sqrt(2 ** self.mu))
        eta = tf.constant(100.0, dtype=tf.float64)
        y = - (m - 1)
        for i in range(1, m):
            y += tf.math.tanh(eta * (x - (2 * i - m) / self.es)) + 1
        y /= self.es
        return y