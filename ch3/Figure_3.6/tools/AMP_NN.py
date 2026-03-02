# -*- coding:utf-8 -*-
# @Time  : 2022/9/10 22:39
# @Author: STARain
# @File  : AMP_NN.py

from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir
import numpy as np
import sys
import tensorflow.compat.v1 as tf
import scipy.io as sio

tf.disable_v2_behavior()
from tensorflow.keras.layers import Dense, GRUCell

def train_amp_nn(test=False, trainset=None):
    Mr, Nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size
    savefile = trainset.savefile
    directory = './model/' + 'AMP_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    model = AMP_NN(trainset=trainset)
    prob = trainset.prob
    x_, y_, H_, sigma2_, label_, bs_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.label_, prob.sample_size_
    xhat, loss_, mse = model.build(x_, y_, H_, sigma2_, label_,
                                              bs=bs_, test=test)  # transfer place holder and build the model

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
    _, _, _, _, _, yval, xval, Hval, sigma2val, labelval = sample_gen(trainset, 1, vsample_size, label=True)
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
                labelbatch = labelval[m * val_batch_size:(m+1) * val_batch_size]
                loss_batch = sess.run(loss_, feed_dict={x_: xbatch, y_: ybatch, H_: Hbatch,
                                                        sigma2_: sigma2batch,
                                                        label_: labelbatch, bs_: val_batch_size})
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
        y, x, H, sigma2, label, _, _, _, _, _ = sample_gen(trainset, batch_size * total_batch, 1, label=True)
        label = label.astype(np.float64)
        # sigma2 = Nt / Mr * 10 ** (-SNR / 10)
        for m in range(total_batch):
            xbatch = x[m * batch_size:(m + 1) * batch_size]
            ybatch = y[m * batch_size:(m + 1) * batch_size]
            Hbatch = H[m * batch_size:(m + 1) * batch_size]
            sigma2batch = sigma2[m * batch_size:(m + 1) * batch_size]
            labelbatch = label[m * batch_size:(m + 1) * batch_size]
            train_loss, _, grad = sess.run((loss_, train, grads_), feed_dict={y_: ybatch, x_:xbatch, H_: Hbatch,
                                                                              sigma2_: sigma2batch,
                                                                              label_: labelbatch,
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


class AMP_NN:
    def __init__(self, trainset=None):
        self.mr, self.nt, self.mu = trainset.m, trainset.n, trainset.mu
        self.m = int(np.sqrt(2 ** self.mu))  # 4 for 16-QAM
        self.dqam = np.sqrt(3 / 2 / (2 ** self.mu - 1))
        self.es = 1 / self.dqam
        if self.mu == 2:
            self.constellation_norm = np.array([+1, -1]) * self.dqam
        elif self.mu == 4:
            self.constellation_norm = np.array([+3, +1, -1, -3]) * self.dqam
        else:
            self.constellation_norm = np.array([+7, +5, +3, +1, -1, -3, -5, -7]) * self.dqam
        self.loss = trainset.loss
        self.iter = trainset.amp_layer
        # trainable variables
        self.dnn = DNN(label_size=2 ** (self.mu // 2), T=0)
        self.constellation_norm = tf.convert_to_tensor(self.constellation_norm)

    def build(self, x, y, A, noise_var, label=tf.zeros([1], tf.float64), bs=None, test=False):
        # convert complex into real
        x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=1)  # (bs, 2nt, 1)
        y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=1)  # (bs, 2nr, 1)
        A = tf.concat([tf.concat([tf.math.real(A), -tf.math.imag(A)], axis=2),
                       tf.concat([tf.math.imag(A), tf.math.real(A)], axis=2)], axis=1)  # (bs, 2nr, 2nt)
        noise_var = noise_var / 2

        # initialization
        ATA = tf.matmul(tf.transpose(A, perm=[0, 2, 1]), A)
        sqrA = A * A
        ave, var = tf.zeros_like(x, tf.float64), self.nt / self.mr * tf.ones_like(x, tf.float64)  # variance initialized as N/M
        w, v = y, tf.ones_like(x, tf.float64)
        ones = np.ones((1, 2 * self.nt, 2 ** (self.mu // 2)), dtype=np.float64)
        eps = tf.constant(5e-13, dtype=tf.float64)

        loss = 0.
        beta = 0.5  # todo: use damping
        mse = []
        vari_feat_vec, hidden_state = [], []
        for t in range(self.iter):
            w = tf.matmul(A, ave) - tf.matmul(sqrA, var) * (y - w) / (
                        noise_var + v)  # note: nominator vt, denominator vt-1
            v = tf.matmul(sqrA, var)
            s = 1 / tf.matmul(tf.transpose(sqrA, perm=[0, 2, 1]), 1 / (noise_var + v))
            r = ave + s * tf.matmul(tf.transpose(A, perm=[0, 2, 1]), (y - w) / (noise_var + v))

            # adjust posterior (cavity) probability distribution using NN
            attr = tf.concat([r, s], axis=2)  # (bs, 2nt, 2)
            loss, p_nn, vari_feat_vec, hidden_state = self.dnn(label, x, y, A, ATA, noise_var, bs=bs,
                                                               u=vari_feat_vec, g_prev=hidden_state, attr=attr, T=t)
            p_nn = tf.transpose(p_nn, perm=[0, 2, 1])  # (bs, 2nt, 2**(mu//2))

            ave = tf.reduce_sum(p_nn * self.constellation_norm, axis=2, keepdims=True)
            var = tf.reduce_sum(p_nn * tf.abs(ave * ones - self.constellation_norm) ** 2, axis=2, keepdims=True)
            var = tf.maximum(var, eps)

            # ave, var, _ = nle(self.mu, r, s, tf.constant(1e-100, dtype=tf.float64))

            mse.append(tf.reduce_mean(tf.square(x - ave)))

        if self.loss == 'mse':
            loss = mse[self.iter - 1]
        ave = tf.complex(ave[:, :self.nt, :], ave[:, self.nt:, :])
        return ave, loss, tf.concat([mse], axis=0)


class MLP(tf.Module):
    def __init__(self, n_hidden_1=64, n_hidden_2=32, output_size=8, activation=None):
        super(MLP, self).__init__()
        # Dense: computes the dot product between the inputs and the kernel
        # along the last axis of the inputs and axis 0 of the kernel
        self.dense1 = Dense(n_hidden_1, activation='relu')
        self.dense2 = Dense(n_hidden_2, activation='relu')
        self.output_layer = Dense(output_size, activation=activation)

    def __call__(self, inputs):
        tmp = self.dense1(inputs)
        tmp = self.dense2(tmp)
        return self.output_layer(tmp)


class DNN:
    def __init__(self, n_hidden_1=32, n_hidden_2=16 , n_gru_hidden_units=32, msg_size=8, label_size=4, T=0):
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.msg_size = msg_size
        self.n_gru_hidden_units = n_gru_hidden_units
        self.label_size = label_size

        # initialize single layer NN
        if T == 0:
            self.W1 = tf.Variable(tf.random.normal([3 + 2, msg_size], dtype=tf.float64), name='W1', trainable=True)
            self.b1 = tf.Variable(tf.zeros([msg_size], dtype=tf.float64), name='b1', trainable=True)

        self.W2 = tf.Variable(tf.random.normal([n_gru_hidden_units, msg_size], dtype=tf.float64),
                              name='W2' + str(T), trainable=True)
        self.b2 = tf.Variable(tf.zeros([msg_size], dtype=tf.float64), name='b2' + str(T), trainable=True)
        # GRU  usage: https://tensorflow.google.cn/guide/keras/rnn
        self.gru = GRUCell(units=n_gru_hidden_units)  # process a single timestep
        # (bs, input_size) --> (bs, output_size/state_size)

        self.mlp = MLP(n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2, output_size=label_size, activation='softmax')

    def __call__(self, label, x, y, h_matrix, HTH, sigma2, u, g_prev, attr, bs=None, L=2, T=0):
        K = h_matrix.shape[2]
        self.bs = bs

        # initial variable feature u for each variable
        if T == 0:
            ydoth = tf.matmul(tf.transpose(y, perm=[0, 2, 1]), h_matrix)  # (bs, 1, K)
            hdoth = tf.expand_dims(tf.linalg.diag_part(HTH), axis=1)  # (bs, 1, K)
            sigma2_vector = sigma2 * tf.ones_like(ydoth, dtype=tf.float64)  # (bs, 1, K)
            chan_info = tf.concat([ydoth, hdoth, sigma2_vector], axis=1)  # (bs, 3, K)
            chan_info = tf.concat([chan_info, tf.transpose(attr, perm=[0, 2, 1])], axis=1)  # (bs, 5, K)
            u = tf.matmul(tf.transpose(chan_info, perm=[0, 2, 1]), self.W1)  # (bs, K, msg_size)
            u = tf.transpose(u + self.b1, perm=[0, 2, 1])  # (bs, msg_size, K)
            g_prev = tf.zeros([bs * K, self.n_gru_hidden_units], dtype=tf.float64)  # (bs * K, n_gru_hidden_units)

        input = tf.concat([tf.reshape(tf.transpose(u, perm=[0, 2, 1]), [bs * K, self.msg_size]),
                           tf.reshape(attr, [bs * K, 2])], axis=1)  # (bs*K, msg_size + attribute_size)
        _, g = self.gru(inputs=input, states=g_prev)  # (bs*K, n_gru_hidden_units)
        g = tf.cast(g, dtype=tf.float64)
        u = tf.matmul(g, self.W2) + self.b2  # (bs*K, msg_size)
        u = tf.transpose(tf.reshape(u, [bs, K, self.msg_size]), perm=[0, 2, 1])  # (bs, msg_size, K)

        # use the feature from the previous layer
        ut = tf.reshape(tf.transpose(u, perm=[0, 2, 1]), [bs * K, self.msg_size])  # (bs*K, msg_size)
        p_nn = self.mlp(ut)  # (bs*K, label_size)
        p_nn = tf.transpose(tf.reshape(p_nn, [bs, K, self.label_size]),
                            perm=[0, 2, 1])  # reverse operation: (bs, label_size, K)
        loss = - tf.reduce_sum(label * tf.math.log(p_nn)) / tf.cast(bs, dtype=tf.float64)
        return loss, p_nn, u, g


sq2 = np.sqrt(2)
sq10 = np.sqrt(10)
sq42 = np.sqrt(42)


def nle(mu, mean, var, thre):
    # ext_probs = np.zeros((ule.shape[0], 2 ** (mu // 2)))
    if mu == 2:  # {-1,+1}
        P0 = tf.maximum(tf.exp(-tf.square(-1 / sq2 - mean) / (2 * var)), thre)  # (bs, 2N, 1)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq2 - mean) / (2 * var)), thre)
        u_post = (P1 - P0) / (P1 + P0) / sq2
        v_post = (P0 * tf.square(u_post + 1 / sq2) + P1 * tf.square(u_post - 1 / sq2)) / (P1 + P0)
        ext_probs = tf.concat([P0, P1], axis=2)
    elif mu == 4:  # {-3,-1,+1,+3}
        P_3 = tf.maximum(tf.exp(-tf.square(-3 / sq10 - mean) / (2 * var)), thre)
        P_1 = tf.maximum(tf.exp(-tf.square(-1 / sq10 - mean) / (2 * var)), thre)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq10 - mean) / (2 * var)), thre)
        P3 = tf.maximum(tf.exp(-tf.square(3 / sq10 - mean) / (2 * var)), thre)
        u_post = (-3 * P_3 - P_1 + P1 + 3 * P3) / (P_3 + P_1 + P1 + P3) / sq10
        v_post = (P_3 * tf.square(u_post + 3 / sq10) + P_1 * tf.square(u_post + 1 / sq10) +
                  P1 * tf.square(u_post - 1 / sq10) + P3 * tf.square(u_post - 3 / sq10)) / (P_3 + P_1 + P1 + P3)
        ext_probs = tf.concat([P_3, P_1, P3, P1], axis=2)  # order corresponds to mapping
    else:  # {-1,+1}
        P_7 = tf.maximum(tf.exp(-tf.square(-7 / sq42 - mean) / (2 * var)), thre)
        P_5 = tf.maximum(tf.exp(-tf.square(-5 / sq42 - mean) / (2 * var)), thre)
        P_3 = tf.maximum(tf.exp(-tf.square(-3 / sq42 - mean) / (2 * var)), thre)
        P_1 = tf.maximum(tf.exp(-tf.square(-1 / sq42 - mean) / (2 * var)), thre)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq42 - mean) / (2 * var)), thre)
        P3 = tf.maximum(tf.exp(-tf.square(3 / sq42 - mean) / (2 * var)), thre)
        P5 = tf.maximum(tf.exp(-tf.square(5 / sq42 - mean) / (2 * var)), thre)
        P7 = tf.maximum(tf.exp(-tf.square(7 / sq42 - mean) / (2 * var)), thre)
        u_post = (-7 * P_7 - 5 * P_5 - 3 * P_3 - P_1 + P1 + 3 * P3 + 5 * P5 + 7 * P7) / (
                P_7 + P_5 + P_3 + P_1 + P1 + P3 + P5 + P7) / sq42
        v_post = (P_7 * tf.square(u_post + 7 / sq42) + P_5 * tf.square(u_post + 5 / sq42) +
                  P_3 * tf.square(u_post + 3 / sq42) + P_1 * tf.square(u_post + 1 / sq42) +
                  P1 * tf.square(u_post - 1 / sq42) + P3 * tf.square(u_post - 3 / sq42) +
                  P5 * tf.square(u_post - 5 / sq42) + P7 * tf.square(u_post - 7 / sq42)) / \
                 (P_7 + P_5 + P_3 + P_1 + P1 + P3 + P5 + P7)
        ext_probs = tf.concat([P_7, P_5, P_1, P_3, P7, P5, P1, P3], axis=2)
    return u_post, v_post, ext_probs