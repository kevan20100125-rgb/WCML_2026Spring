#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
import numpy as np
import sys
import tensorflow.compat.v1 as tf
import scipy.io as sio

tf.disable_v2_behavior()

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
    prob = trainset.prob
    x_, y_, H_, sigma2_, bs_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.sample_size_
    model = CGNet(trainset=trainset)
    H_fixed = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt) +
                                   1j * np.random.randn(Mr, Nt))
    xhat, loss_, mse, layers = model.build(x_, y_, H_, sigma2_,
                                           bs=bs_, test=test)  # transfer place holder and build the model

    train, grads_ = [], []
    total_var_list = tuple()
    global_step = tf.Variable(0, trainable=False)
    lr_ = tf.train.exponential_decay(lr, global_step, decay_steps, lr_decay, name='lr')

    if test is False:
        # grads_, _ = tf.clip_by_global_norm(tf.gradients(loss_, tf.trainable_variables()),
        #                                    trainset.grad_clip)
        if trainset.grad_clip_flag:
            optimizer = tf.train.AdamOptimizer(lr_)
            # grads_, _ = tf.clip_by_global_norm(tf.gradients(loss_, tf.trainable_variables()),
            #                                    trainset.grad_clip)
            if tf.trainable_variables():
                train = optimizer.apply_gradients(zip(grads_, tf.trainable_variables()), global_step)
        else:
            for name, xhat_, var_list, mse in layers:
                if var_list is not None:
                    train_ = tf.train.AdamOptimizer(lr_).minimize(mse, global_step, var_list=var_list)
                    train.append((name, xhat_, mse, train_, var_list))
                    total_var_list = total_var_list.__add__(var_list)
                    train_refine = tf.train.AdamOptimizer(lr_ * .5).minimize(mse, global_step,
                                                                             var_list=total_var_list)
                    train.append((name + ' fine-tune', xhat_, mse, train_refine, total_var_list))

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
    better_wait = 500
    # generate validation set
    _, _, _, _, yval, xval, Hval, sigma2val = sample_gen(trainset, 1, vsample_size,
                                                         fixed_channel=False, H_fixed=H_fixed)
    # sigma2val = Nt / Mr * 10 ** (-SNR / 10)
    val_batch_size = vsample_size // total_batch


    for name, xhat_, loss_, train_, var_list in train:
        name_SNR = name + ' with SNR={SNR: .6f}'.format(SNR=SNR)
        if name_SNR in done:
            print('Already did ' + name_SNR + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])
        print(name_SNR + ' ' + describe_var_list)
        loss_history = []
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
                    age_of_best = len(loss_history) - loss_history.argmin() - 1
                    if age_of_best * ivl > better_wait:
                        break

            # generate trainset
            y, x, H, sigma2, _, _, _, _ = sample_gen(trainset, batch_size * total_batch, 1,
                                                     fixed_channel=False, H_fixed=H_fixed)
            # sigma2 = Nt / Mr * 10 ** (-SNR / 10)
            for m in range(total_batch):
                train_loss, _ = sess.run((loss_, train_),
                                               feed_dict={y_: y[m * batch_size:(m + 1) * batch_size],
                                                          x_: x[m * batch_size:(m + 1) * batch_size],
                                                          H_: H[m * batch_size:(m + 1) * batch_size],
                                                          sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                                          bs_: batch_size})

        done = np.append(done, name_SNR)
        log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                    'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin() * ivl)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess, savefile, **state)

    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k)
            # print('restoring ' + k + ' = ' + str(d))
    save_trainable_vars(sess, savefile, **state)

    para = {}
    for k, v in np.load(savefile).items():
        para[k.replace(':', '').replace('/', '_')] = v
    sio.savemat(savefile.replace('.npz', '.mat'), para)

    return


class CGNet:
    def __init__(self, trainset=None):
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
            # self.alpha.append(tf.Variable(0.0, dtype=tf.float64, name='alpha_' + str(t), trainable=True))
            self.alpha.append(tf.Variable(np.zeros((self.nt, 1)), dtype=tf.float64,
                                          name='alpha_' + str(t), trainable=True))
            if t != self.iter - 1:
                # self.beta.append(tf.Variable(0.0, dtype=tf.float64, name='beta_' + str(t), trainable=True))
                self.beta.append(tf.Variable(np.zeros((self.nt, 1)), dtype=tf.float64,
                                             name='beta_' + str(t), trainable=True))

    def build(self, x, y, A, noise_var, bs=None, test=False):
        xhat = tf.zeros_like(x, tf.complex128)
        noise_var = tf.cast(noise_var, dtype=tf.complex128)
        AH = tf.linalg.adjoint(A)
        AHA = tf.matmul(AH, A)
        ytilde = tf.matmul(AH, y)
        xi = AHA + noise_var * tf.eye(self.nt, batch_shape=[bs], dtype=tf.complex128)
        r = ytilde - tf.matmul(xi, xhat)  # (bs, nt, 1)
        di = r

        loss = 0.
        mse = []
        layers = []
        for i in range(self.iter):
            # compute the approximate solution based on prior conjugate direction and residual
            xi_di = tf.matmul(xi, di)
            xhat = xhat + tf.cast(self.alpha[i], dtype=tf.complex128) * di

            mse.append((tf.nn.l2_loss(tf.math.real(xhat) - tf.math.real(x)) +
                        tf.nn.l2_loss(tf.math.imag(xhat) - tf.math.imag(x)))
                       / tf.cast(bs * self.nt / 2, dtype=tf.float64))
            if i == self.iter - 1:
                layers.append(('CGNet T={0}'.format(i), xhat, (self.alpha[i],), mse[i]))
                continue
            layers.append(('CGNet T={0}'.format(i), xhat, (self.alpha[i], self.beta[i]), mse[i]))
            # compute conjugate direction and residual
            r -= tf.cast(self.alpha[i], dtype=tf.complex128) * xi_di
            di = r + tf.cast(self.beta[i], dtype=tf.complex128) * di

        if self.loss == 'mse':
            loss += mse[self.iter - 1]

        return xhat, loss, tf.concat([mse], axis=0), layers


