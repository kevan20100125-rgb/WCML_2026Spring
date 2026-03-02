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


def train_fs_net(test=False, trainset=None):
    Mr, Nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size
    savefile = trainset.savefile
    directory = './model/' + 'FS_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainset.prob
    x_, y_, H_, sigma2_, bs_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.sample_size_
    model = FSNet(trainset=trainset)

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


class FSNet:
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
        self.constellation_norm = tf.convert_to_tensor(self.constellation_norm)
        self.loss = trainset.loss
        self.iter = trainset.fs_layer
        # trainable variables
        self.weight, self.bia = [], []
        for t in range(self.iter):
            self.weight.append(tf.Variable(tf.random_normal([4 * self.nt, 1], stddev=0.7, dtype=tf.float64),
                                            name='weight_' + str(t), trainable=True))
            self.bia.append(tf.Variable(tf.zeros([4 * self.nt, 1], dtype=tf.float64),
                                         name='bias_' + str(t), trainable=True))

    def build(self, x, y, A, noise_var, bs=None, test=False):
        # convert complex into real
        x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=1)  # (bs, 2nt, 1)
        y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=1)  # (bs, 2nr, 1)
        A = tf.concat([tf.concat([tf.math.real(A), -tf.math.imag(A)], axis=2),
                       tf.concat([tf.math.imag(A), tf.math.real(A)], axis=2)], axis=1)  # (bs, 2nr, 2nt)
        xhat = tf.zeros_like(x, tf.float64)

        x_tran = tf.transpose(x, perm=[0, 2, 1])
        x_norm = tf.norm(x, axis=1, keepdims=True)  # (bs, 1, 1)
        AT = tf.transpose(A, perm=[0, 2, 1])
        ATA = tf.matmul(AT, A)  # todo: can be reused in MHGD
        ytilde = tf.matmul(AT, y)

        loss = 0.
        beta = 0.5
        mse = []
        x_hat = None  # debug
        for t in range(self.iter):
            input_vector = tf.concat([xhat, tf.matmul(ATA, xhat) - ytilde], axis=1)  # (bs, 4nt, 1)
            tmp = self.weight[t] * input_vector + self.bia[t]  # (bs, 4nt, 1)
            tmp = tmp[:, :2 * self.nt, :] + tmp[:, 2 * self.nt:, :]  # (bs, 2nt, 1)
            xhat = self.piecewise_linear_soft_sign(tmp, iter=t)  # (bs, 2nt, 1)
            correlation = tf.reduce_mean(abs(tf.matmul(x_tran, xhat)) / (x_norm * tf.norm(xhat, axis=1, keepdims=True)))
            mse.append(tf.reduce_mean(tf.square(x - xhat)))
            # correlation: 0-1; mse (for each symbol): -20 dB ~ 0.01; \times 2nt
            loss += np.log(t + 2) * (mse[t] + beta * (1 - correlation))
            # loss += np.log(t + 2) * (mse[t])
            if t == self.iter - 2:
                x_hat = xhat

        if test:
            indices = tf.argmin(abs(xhat * np.ones((1, 2 * self.nt, self.m)) - self.constellation_norm),
                                axis=2)  # (bs, 2 * nt)
            x_qua = tf.gather_nd(self.constellation_norm, indices[:, :, tf.newaxis])[:, :, tf.newaxis]  # (bs, 2nt, 1)
            x_qua = tf.complex(x_qua[:, :self.nt, :], x_qua[:, self.nt:, :])
        else:
            x_qua = self.approx_quantization(xhat)

        return x_qua, loss, tf.concat([mse], axis=0)

    def piecewise_linear_soft_sign(self, x, iter):
        t = tf.Variable(0.1, dtype=tf.float64, trainable=True, name='t_' + str(iter))  # todo: a good value?
        m = self.m  # 4 for 16-QAM
        y = - (m - 1)  # -3 for 16-QAM
        for i in range(1, m):
            bias = (2 * i - m) / self.es  # {-2, 0, 2} / sqrt(10) for 16-QAM
            y += (tf.nn.relu(x + bias + t) - tf.nn.relu(x + bias - t)) / abs(t + 0.00001)
        y /= self.es
        return y

    def approx_quantization(self, x):
        m = self.m
        eta = tf.constant(100.0, dtype=tf.float64)
        y = - (m - 1)
        for i in range(1, m):
            y += tf.math.tanh(eta * (x - (2 * i - m) / self.es)) + 1
        y /= self.es
        return y
