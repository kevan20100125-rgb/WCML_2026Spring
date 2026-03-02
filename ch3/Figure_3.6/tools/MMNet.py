
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import scipy.io as sio
tf.disable_v2_behavior()
from .utils import mkdir
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen


def train_MMNet(test=False, trainSet=None):
    Mr, Nt, mu, SNR = trainSet.m, trainSet.n, trainSet.mu, trainSet.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainSet.lr, trainSet.lr_decay, trainSet.decay_steps, trainSet.min_lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    directory = './model/' + 'MMNet_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainSet.prob
    x_, y_, H_, sigma2_, label_, bs_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.label_, prob.sample_size_
    model = MMNet(trainSet)
    xhat, loss_ = model.build(x_, y_, H_, sigma2_, bs=bs_)

    train = []
    global_step = tf.Variable(0, trainable=False)
    lr_ = tf.train.exponential_decay(lr, global_step, decay_steps, lr_decay, name='lr')
    grads_, _ = tf.clip_by_global_norm(tf.gradients(loss_, tf.trainable_variables()),
                                       trainSet.grad_clip)
    if trainSet.grad_clip_flag:
        optimizer = tf.train.AdamOptimizer(lr_)
        if tf.trainable_variables():
            train = optimizer.apply_gradients(zip(grads_, tf.trainable_variables()), global_step)
    else:
        if tf.trainable_variables():
            train = tf.train.AdamOptimizer(lr_).minimize(loss_, global_step, var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)

    if test:
        return sess, xhat[-1]

    loss_history = []
    save = {}  # for the best model
    ivl = 5
    # generate validation set
    _, _, _, _, _, _, yval, xval, Hval, sigma2val, labelval, rw_inv_val = sample_gen(trainSet, 1, vsample_size)
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
            y, x, H, sigma2, label, rw_inv, _, _, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)
            # sigma2 = Nt / Mr * 10 ** (-SNR / 10)
            for m in range(total_batch):
                train_loss, _, grad = sess.run((loss_, train, grads_),
                                               feed_dict={y_: y[m * batch_size:(m + 1) * batch_size],
                                                          x_: x[m * batch_size:(m + 1) * batch_size],
                                                          H_: H[m * batch_size:(m + 1) * batch_size],
                                                          sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                                          bs_: batch_size})
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


class MMNet:
    def __init__(self, trainset):
        self.mr, self.nt, self.mu = trainset.m, trainset.n, trainset.mu
        self.L = trainset.T
        if self.mu == 2:
            self.constel_norm = np.array([+1, -1]) / np.sqrt(2)
        elif self.mu == 4:
            self.constel_norm = np.array([+3, +1, -1, -3]) / np.sqrt(10)
        else:
            self.constel_norm = np.array([+7, +5, +3, +1, -1, -3, -5, -7]) / np.sqrt(42)
        self.M = int(self.constel_norm.shape[0])

    def build(self, x, y, H, noise_var, bs=None):
        xhatk = tf.zeros(shape=[bs, 2 * self.nt])
        xhat = []
        for k in range(1, self.L):
            zt, rt, W = self.linear(H, y, xhatk)
            xhatk = self.denoiser(H, W, zt, rt, noise_var, k)
            xhat.append(xhatk)
        loss = 0.
        for xhatk in xhat:
            lk = tf.losses.mean_squared_error(labels=x, predictions=xhatk)
            loss += lk
            tf.add_to_collection(tf.GraphKeys.LOSSES, lk)
        return xhat, loss

    def linear(self, H, y, xhat):
        rt = y - self.batch_matvec_mul(H, xhat)
        W = tf.matrix_transpose(H)
        zt = xhat + self.batch_matvec_mul(W, rt)
        return zt, rt, W

    def denoiser(self, H, W, zt, rt, noise_sigma, t):
        '''rt: (bs, 2nt)'''
        HTH = tf.matmul(H, H, transpose_a=True)  # (bs, 2nt, 2nt)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - 2 * self.mr * tf.square(noise_sigma) / 2.),
                         tf.expand_dims(tf.trace(HTH), axis=1))  # (bs, 2nt)
        v2_t = tf.maximum(v2_t, 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)  # (bs, 2nt, 1)
        C_t = tf.eye(2 * self.nt, batch_shape=[tf.shape(H)[0]]) - tf.matmul(W, H)
        tau2_t = 1 / self.nt / 2 * tf.reshape(tf.trace(tf.matmul(C_t, C_t, transpose_b=True)), [-1,1,1]) * v2_t + \
                 tf.square(tf.reshape(noise_sigma, [-1, 1, 1])) / (4 * self.nt) * tf.reshape(tf.trace(tf.matmul(W, W, transpose_b=True)), [-1,1,1])
        xhat = self.gaussian(zt, tau2_t / tf.Variable(tf.random_normal([1, 2*self.nt, 1], 1., 0.1),
                                              name='theta_' + str(t), trainable=True))  # trainable parameters for MMNet-iid
        return xhat

    def gaussian(self, zt, tau2_t):
        # zt - symbols
        arg = tf.reshape(zt, [-1, 1]) - self.constel_norm
        arg = tf.reshape(arg, [-1, 2 * self.nt, self.M])
        # -|| z - symbols||^2 / 2sigma^2
        arg = - tf.square(arg) / 2. / tau2_t
        arg = tf.reshape(arg, [-1, self.M])
        x_out = tf.nn.softmax(arg, axis=1)
        # sum {xi exp()/Z}
        x_out = tf.matmul(x_out, tf.reshape(self.constel_norm, [self.M, 1]))
        x_out = tf.reshape(x_out, [-1, 2 * self.nt])  # (bs, 2nt)
        return x_out

    def batch_matvec_mul(self, A, b, transpose_a=False):
        '''Multiplies a matrix A of size batch_sizexNxK
           with a vector b of size batch_sizexK
           to produce the output of size batch_sizexN
        '''
        C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
        return tf.squeeze(C, -1)
