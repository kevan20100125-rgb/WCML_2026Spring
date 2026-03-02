#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
# from .MIMO_detection import sample_gen
from .utils import mkdir
import numpy as np
import numpy.linalg as la
import sys
import time
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import scipy.io as sio
# import tensorflow as tf

# from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GRUCell


def train_GEPNet(test=False, trainSet=None):
    T, iter = trainSet.T, trainSet.TurboIterations - 1
    Mr, Nt, mu, SNR = trainSet.m, trainSet.n, trainSet.mu, trainSet.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainSet.lr, trainSet.lr_decay, trainSet.decay_steps, trainSet.min_lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    tsample_size = total_batch * batch_size
    channel_type = trainSet.channel_type
    savefile = trainSet.savefile
    directory = './model/' + 'GEPNet_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainSet.prob
    x_, y_, H_, sigma2_, label_, bs_, rw_inv_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.label_,\
                                                prob.sample_size_, prob.rw_inv_
    model = GEPNet(trainSet=trainSet)
    loss_, ub, cavity_prob, mse = model.build(x_, y_, H_, sigma2_, label_,
                                              bs=bs_, rw_inv=rw_inv_)  # transfer place holder and build the model
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
    # layers = []
    # for t in range(T):
    #     layers.append(('EP T={0}'.format(t), (model.trainable_variables[t],)))
    #
    # for name, var_list in layers:
    #     if len(var_list):
    #         describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
    #     else:
    #         describe_var_list = 'fine tuning all ' + ','.join([v.name for v in model.trainable_variables])
    #     done = np.append(done, name)
    #     print(name + ' ' + describe_var_list)
    if test:
        return sess, ub, cavity_prob, mse

    if channel_type == '3gpp':
        if Nt == 16:
            Hdataset = sio.loadmat('../MCMC/data/H_3gpp_64x16.mat')['H1']
        elif Nt == 8:
            Hdataset = sio.loadmat('../MCMC/data/H_3gpp_20mhz.mat')['H1']
        trainSet.Hdataset = Hdataset

    loss_history = []
    save = {}  # for the best model
    ivl = 5
    # generate validation set
    _, _, _, _, _, _, yval, xval, Hval, sigma2val, labelval, rw_inv_val = sample_gen(trainSet, 1, vsample_size)
    # sigma2val = Nt / Mr * 10 ** (-SNR / 10)
    val_batch_size = vsample_size // total_batch
    # loss_batch, grads = grad(xval, yval, Hval, sigma2val, labelval)  # call to generate all trainable variables

    for i in range(maxit + 1):
        if i % ivl == 0:  # validation:don't use optimizer
            loss = 0.
            for m in range(total_batch):
                xbatch = xval[m * val_batch_size: (m + 1) * val_batch_size]
                ybatch = yval[m * val_batch_size: (m + 1) * val_batch_size]
                Hbatch = Hval[m * val_batch_size:(m + 1) * val_batch_size]
                sigma2batch = sigma2val[m * val_batch_size:(m + 1) * val_batch_size]
                labelbatch = labelval[m * val_batch_size:(m + 1) * val_batch_size]
                rw_inv_batch = rw_inv_val[m * val_batch_size:(m + 1) * val_batch_size]
                loss_batch = sess.run(loss_, feed_dict={y_: ybatch,
                                                        x_: xbatch, H_: Hbatch, sigma2_: sigma2batch,
                                                        bs_: val_batch_size,
                                                        label_: labelbatch,
                                                        rw_inv_: rw_inv_batch})
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
            sta = time.time()
            train_loss, _, grad = sess.run((loss_, train, grads_),
                                           feed_dict={y_: y[m * batch_size:(m + 1) * batch_size],
                                                      x_: x[m * batch_size:(m + 1) * batch_size],
                                                      H_: H[m * batch_size:(m + 1) * batch_size],
                                                      sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                                      bs_: batch_size,
                                                      label_: label[m * batch_size:(m + 1) * batch_size],
                                                      rw_inv_: rw_inv[m * batch_size:(m + 1) * batch_size]})
            # if grad > 100.0:
            #     pass
            time_train = time.time() - sta

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


class GEPNet:  # todo: test the revised of initialization
    def __init__(self, trainSet=None):
        self.T, iter = trainSet.T, trainSet.TurboIterations - 1
        self.Nt, self.mu = trainSet.n, trainSet.mu
        # self.GNN = []
        size = trainSet.size
        self.torch = trainSet.torch
        self.coop = trainSet.coop
        self.GNN = GNN_GRU(n_hidden_1=size, n_hidden_2=size / 2, n_gru_hidden_units=size,
                           label_size=2 ** (self.mu // 2), T=0, ed=trainSet.ed, ed_para=trainSet.ed_para,
                           torch=self.torch)
        if self.mu == 2:
            self.real_constel_norm = np.array([+1, -1]) / np.sqrt(2)
        elif self.mu == 4:
            self.real_constel_norm = np.array([+3, +1, -1, -3]) / np.sqrt(10)
        else:
            self.real_constel_norm = np.array([+7, +5, +3, +1, -1, -3, -5, -7]) / np.sqrt(42)
        self.loss = trainSet.loss
        self.modified = trainSet.modified
        # with self.name_scope:
        # for t in range(self.T):
        #     self.GNN.append(GNN_GRU(label_size=2**(self.mu//2), T=t))

    # @tf.Module.with_name_scope
    def build(self, x, y, H, sigma2, label=tf.zeros([1], tf.float64), bs=None, test=False, rw_inv=None):
        HT = tf.transpose(H, perm=[0, 2, 1])
        HTH = tf.matmul(HT, H)
        # sigma2 = tf.cast(sigma2 / 2, dtype=tf.float64)
        noise_var = tf.cast(sigma2 / 2, dtype=tf.float64)

        # precompute some tensorflow constants
        eps1 = tf.constant(1e-7, dtype=tf.float64)
        eps2 = tf.constant(5e-13, dtype=tf.float64)
        pth = tf.constant(1e-100, dtype=tf.float64)
        # Lambda0 = tf.Variable(float(1 / 0.5), dtype=tf.float64, name='Lambda')
        # Lambda = Lambda0 * tf.ones_like(x, dtype=tf.float64)
        Lambda = 1 / (0.5 * tf.ones_like(x, dtype=tf.float64))
        gamma = tf.zeros_like(x, tf.float64)
        beta = tf.constant(0.2, dtype=tf.float64)

        loss = 0.
        p_gnn = 0.
        vari_feat_vec, hidden_state = [], []
        mse = []
        for t in range(self.T):
            # compute the mean and covariance matrix
            # (bs, 2N, 2N)  lambda:(bs, 2N, 1) -> (bs, 2N, 2N)
            if self.modified:
                Sigma = tf.linalg.inv(tf.matmul(tf.matmul(HT, rw_inv), H) + tf.linalg.diag(tf.squeeze(Lambda)))
                Mu = tf.matmul(Sigma, (tf.matmul(HT, tf.matmul(rw_inv, y)) + gamma))
            else:
                Sigma = tf.linalg.inv(HTH + noise_var * tf.linalg.diag(tf.squeeze(Lambda)))  ###
                Mu = tf.matmul(Sigma, tf.matmul(HT, y) + noise_var * gamma)  # (bs, 2N, 1)   ###

            # compute the extrinsic mean and covariance matrix
            if self.modified:
                diag = tf.expand_dims(tf.linalg.diag_part(Sigma), -1)
            else:
                diag = noise_var * tf.expand_dims(tf.linalg.diag_part(Sigma), -1)  # (bs, 2N, 2N) -> (bs, 2N, 1)###
            vab = tf.divide(diag, 1 - diag * Lambda)  # (bs, 2N, 1)  ###
            vab = tf.maximum(vab, eps1)
            uab = vab * (Mu / diag - gamma)  # (bs, 2N, 1)
            attr = tf.concat([uab, vab], axis=2)  # (bs, 2N, 2)

            # adjust posterior (cavity) probability distribution using GNN
            # correlation coefficient of x: rho = Sigma.^2 ./ diag(Sigma) ./ diag(Sigma).T
            diag = diag / noise_var
            rho = Sigma ** 2 / diag / tf.transpose(diag, perm=[0, 2, 1])
            gnn_loss, p_gnn, vari_feat_vec, hidden_state = self.GNN(label, x, y, H, - HTH if self.torch else HTH,
                                                                    noise_var, bs=bs,
                                                                    u=vari_feat_vec, g_prev=hidden_state,
                                                                    attr=attr, T=t, cov=rho)
            # _, _, ext_probs = nle(self.mu, uab, vab, pth)
            # compute the posterior mean and covariance matrix
            # vab = tf.cast(vab, dtype=tf.float64)
            # uab = tf.cast(uab, dtype=tf.float64)
            p_gnn = tf.transpose(p_gnn, perm=[0, 2, 1])
            ub = tf.reduce_sum(p_gnn * self.real_constel_norm,
                               axis=2, keepdims=True)
            vb = tf.reduce_sum(p_gnn * tf.abs(ub * np.ones((1, x.shape[1], 2 ** (self.mu // 2)), dtype=np.float64)
                                              - self.real_constel_norm) ** 2, axis=2, keepdims=True)
            vb = tf.maximum(vb, eps2)
            mse.append(tf.reduce_mean((ub - x) ** 2))
            # ub = tf.cast(ub, dtype=tf.float32)
            # vb = tf.maximum(tf.cast(vb, dtype=tf.float32), eps)
            # vab = tf.cast(vab, dtype=tf.float32)
            # uab = tf.cast(uab, dtype=tf.float32)

            # moment matching and damping
            gamma_last = gamma
            Lambda_last = Lambda
            gamma = (ub * vab - uab * vb) / vb / vab  ###
            Lambda = (vab - vb) / vb / vab  ###
            # If x and y are also provided (both have non-None values) the condition tensor acts as a mask
            # that chooses whether the corresponding element / row in the output should be taken from x
            # (if the element in condition is True) or y (if it is False).
            condition = Lambda < 0.
            Lambda = tf.where(condition=condition, x=Lambda_last, y=Lambda)
            gamma = tf.where(condition=condition, x=gamma_last, y=gamma)
            gamma = beta * gamma + (1 - beta) * gamma_last
            Lambda = beta * Lambda + (1 - beta) * Lambda_last
            if self.loss == 'avg':
                loss += gnn_loss / self.T
            elif self.loss == 'weighted':  # larger
                loss += gnn_loss * (t + 1) / self.T
        if self.loss == 'final':
            loss += gnn_loss
        return loss, ub, p_gnn, tf.concat([mse], axis=0)


class mlp(tf.Module):
    def __init__(self, n_hidden_1=64, n_hidden_2=32, output_size=8,
                 activation1=None, activation2=None, activation3=None):
        super(mlp, self).__init__()
        # Dense: computes the dot product between the inputs and the kernel
        # along the last axis of the inputs and axis 0 of the kernel
        self.dense1 = Dense(n_hidden_1, activation=activation1)
        self.dense2 = Dense(n_hidden_2, activation=activation2)
        self.output_layer = Dense(output_size, activation=activation3)

    def __call__(self, inputs, coop=False):
        tmp = self.dense1(inputs)
        tmp = self.dense2(tmp)
        # if coop:
        #     return self.output_layer(tmp), tmp
        return self.output_layer(tmp)


class GNN_GRU:  # todo: implementation of network, size
    # n_hidden_1 = n_gru_hidden_units
    def __init__(self, n_hidden_1=64, n_hidden_2=32, n_gru_hidden_units=64, msg_size=8, label_size=4, T=0,
                 ed=None, ed_para=None, torch=False):
        # with self.name_scope:
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.msg_size = msg_size
        self.n_gru_hidden_units = n_gru_hidden_units
        self.label_size = label_size
        self.ed, self.ed_para = ed, ed_para

        # factor mlp using subclass # input_size: (None, 2*msg_size+3)
        self.factor_mlp = mlp(output_size=msg_size, n_hidden_1=n_hidden_1,
                              n_hidden_2=n_hidden_2, activation1='relu',
                              activation2='relu', activation3='relu' if torch else None)

        # GRU  usage: https://tensorflow.google.cn/guide/keras/rnn
        self.gru = GRUCell(units=n_gru_hidden_units)  # process a single timestep
        # (bs, input_size) --> (bs, output_size/state_size)
        # self.rnn = RNN(self.gru(1), return_state=True)  # wrap into a RNN layer
        # input: (batch_size, n_time_steps, input_size)

        # initialize single layer NN
        if T == 0:
            self.W1 = tf.Variable(tf.random.normal([3, msg_size], dtype=tf.float64, stddev=0.1),
                                  name='W1', trainable=True)
            self.b1 = tf.Variable(tf.zeros([msg_size], dtype=tf.float64), name='b1', trainable=True)

        # variable mlp
        self.W2 = tf.Variable(tf.random.normal([n_gru_hidden_units, msg_size], dtype=tf.float64, stddev=0.1),
                              name='W2' + str(T), trainable=True)
        self.b2 = tf.Variable(tf.zeros([msg_size], dtype=tf.float64), name='b2' + str(T), trainable=True)

        # readout mlp using subclass
        self.readout_mlp = mlp(output_size=label_size, activation1=None if torch else 'relu',
                               activation2=None if torch else 'relu', activation3='softmax',
                               n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2)  # input_size: (None, msg_size)

    # @tf.Module.with_name_scope
    def __call__(self, label, x, y, h_matrix, HTH, sigma2, u, g_prev, attr, bs=None, L=2, T=0,
                 cov=None):
        K = x.shape[1]
        self.bs = bs
        # initial variable feature u for each variable -- variable feature vector
        if T == 0:
            ydoth = tf.matmul(tf.transpose(y, perm=[0, 2, 1]), h_matrix)  # (bs, 1, K)
            hdoth = tf.expand_dims(tf.linalg.diag_part(HTH), axis=1)  # (bs, 1, K)
            sigma2_vector = sigma2 * tf.ones_like(ydoth, dtype=tf.float64)  # (bs, 1, K)
            chan_info = tf.concat([ydoth, hdoth, sigma2_vector], axis=1)  # (bs, 3, K)
            u = tf.matmul(tf.transpose(chan_info, perm=[0, 2, 1]), self.W1)  # (bs, K, msg_size)
            u = tf.transpose(u + self.b1, perm=[0, 2, 1])  # (bs, msg_size, K)
            g_prev = tf.zeros([bs * K, self.n_gru_hidden_units], dtype=tf.float64)  # (bs * K, n_gru_hidden_units=64)
            # g_prev = tf.zeros([u.shape[0], self.n_gru_hidden_units*K])  # (bs, n_gru_hidden_units=64 * K)

        # msg = tf.zeros([y.shape[0], K, K,], dtype=tf.float64)  # (bs, K, K, msg_size)  zero-diagonal
        for l in range(L):
            # factor to variable
            sta = time.time()
            msg = self.factor2variable(K, u, HTH, sigma2, cov=cov)  # (bs, K, msg_size)
            time1 = time.time() - sta
            # variable to factor
            sta2 = time.time()
            u, g_prev = self.variable2factor(K, msg=msg,  # u: (bs, msg_size, K) g: (bs, n_gru_hidden_units*K)
                                             attr=attr, g_prev=g_prev)
            time2 = time.time() - sta2
        # readout & calculate loss
        sta3 = time.time()
        p_gnn = self.readout(K, u)  # (bs, label_size, K)
        p_gnn = tf.cast(p_gnn, dtype=tf.float64)
        time3 = time.time() - sta3
        loss = -tf.reduce_sum(label * tf.math.log(p_gnn)) / tf.cast(bs, dtype=tf.float64)  # todo: cross-entropy loss
        # feedback GRU hidden state and variable feature vector
        return loss, p_gnn, u, g_prev

    def factor2variable(self, K, u, HTH, sigma2, cov=None):  # factor2variable  K(K-1) factors  each factor has a MLP?
        all_input = []
        mask = []
        # flag = tf.ones([self.bs, K - 1, K], dtype=tf.int)
        for k in range(K):
            # msgk = 0.
            # for j in range(K):
            #     if j != k:
            #         factor_feat = tf.squeeze(tf.concat([tf.expand_dims(HTH[:,k,j], axis=-1),  # (bs, 2)
            #                                             sigma2*tf.ones([u.shape[0],1], dtype=tf.float64)], axis=1))
            #         input = tf.concat([u[:,:,k], u[:,:,j], factor_feat], axis=1)  # (bs, 2*msg_size+2)
            #         msgjk = tf.cast(self.factor_mlp(input), dtype=tf.float64)  # (bs, msg_size)
            #         msgk += msgjk
            u_source = tf.expand_dims(u[:, :, k], axis=1) * tf.ones([self.bs, K - 1, u.shape[1]],
                                                                    dtype=tf.float64)  # (bs, K-1, msg_size)
            u_target = tf.transpose(tf.concat([u[:, :, :k], u[:, :, k + 1:]], axis=2),
                                    perm=[0, 2, 1])  # (bs, K-1, msg_size)
            factor_feat = tf.expand_dims(tf.concat([HTH[:, k, :k], HTH[:, k, k + 1:]], axis=1), axis=-1)  # (bs, K-1, 1)
            if self.ed:
                cov_feat = tf.expand_dims(tf.concat([cov[:, k, :k], cov[:, k, k + 1:]], axis=1),
                                          axis=-1)  # (bs, K-1, 1)
                avg_rho = tf.reduce_mean(cov_feat, axis=1, keepdims=True)  # (bs, 1, 1)
                flag = tf.transpose(cov_feat >= self.ed_para * avg_rho, perm=[0, 2, 1])  # (bs, 1, K - 1)
                flag = tf.reshape(tf.repeat(flag, repeats=self.msg_size),
                                  [self.bs, 1, K - 1, self.msg_size])  # (bs, 1, K - 1, msg_size)
                mask.append(flag)
            input = tf.concat([u_source, u_target, factor_feat,
                               sigma2 * tf.ones([self.bs, K - 1, 1], dtype=tf.float64)],
                              axis=2)  # (bs, K-1, 2*msg_size+3)
            all_input.append(input)
        all_input = tf.stack(all_input, axis=1)  # (bs, K, K-1, 2*msg_size+3)
        msg = tf.cast(self.factor_mlp(tf.reshape(all_input, [-1, K, K - 1, 2 * self.msg_size + 2])),
                      dtype=tf.float64)  # (bs, K, K-1, msg_size)
        if self.ed:
            mask = tf.concat(mask, axis=1)  # (bs, K, K-1, msg_size)
            msg = tf.reduce_sum(tf.where(condition=mask, x=msg,
                                         y=tf.zeros([self.bs, K, K - 1, self.msg_size], dtype=tf.float64)), axis=2)
        else:
            msg = tf.reduce_sum(msg, axis=2)  # (bs, K, msg_size)
        # msg = tf.reduce_sum(tf.reshape(tf.boolean_mask(msg, mask), [self.bs, K, self.msg_size]), axis=2)  # (bs, K, msg_size)
        # msgk = tf.cast(self.factor_mlp(input), dtype=tf.float64)  # (bs, K-1, msg_size)
        # msgk = tf.reduce_sum(msgk, axis=1)  # (bs, msg_size)
        # if k == 0:
        #     msg = msgk  # (bs, msg_size)
        # else:
        #     msg = tf.concat([msg, msgk], axis=1)
        return msg

    def variable2factor(self, K, msg, attr, g_prev):  # variable2factor h:hidden vectors  K variables, share a GRU
        # todo: keras implementation of GRU has already concated g_prev? argument in call or automatically transfer?
        msg_size = self.msg_size
        state_size = self.n_gru_hidden_units
        batch_size = self.bs
        msg = tf.reshape(msg, [batch_size * K, msg_size])  # (bs*K, msg_size)
        attr = tf.reshape(attr, [batch_size * K, 2])  # (bs*K, 2)
        input = tf.concat([msg, attr], axis=1)  # (bs*K, msg_size+2)
        # input = msg
        _, g = self.gru(inputs=input, states=g_prev)  # g: (bs*K, n_gru_hidden_units=64)
        g = tf.cast(g, dtype=tf.float64)
        u = tf.matmul(g, self.W2) + self.b2  # (bs*K, msg_size)
        u = tf.transpose(tf.reshape(u, [batch_size, K, msg_size]), perm=[0, 2, 1])  # (bs, msg_size, K)
        # for k in range(K):
        #     input = tf.concat([msg[:,k*msg_size:(k+1)*msg_size], attr[:,k,:]], axis=1)  # (bs, msg_size+2)
        #     # todo: gnn implementation
        #     _, gk = self.gru(inputs=input,
        #                      states=g_prev[:, k*state_size:(k+1)*state_size])  # gk: (bs, n_gru_hidden_units=64)
        #     gk = tf.cast(gk, dtype=tf.float64)
        #     uk = tf.expand_dims(tf.matmul(gk, self.W2) + self.b2, axis=-1)  # (bs, msg_size, 1)
        #     if k == 0:
        #         u = uk
        #         g = gk
        #     else:
        #         u = tf.concat([u, uk], axis=2)
        #         g = tf.concat([g, gk], axis=1)
        # g = tf.cast(g, dtype=tf.float32)  # state should be stored as float32
        return u, g

    def readout(self, K, u):
        batch_size = self.bs
        u = tf.reshape(tf.transpose(u, perm=[0, 2, 1]), [batch_size * K, self.msg_size])  # (bs*K, msg_size)
        p_gnn = self.readout_mlp(u, coop=True)  # (bs*K, label_size)
        p_gnn = tf.transpose(tf.reshape(p_gnn, [batch_size, K, self.label_size]),
                             perm=[0, 2, 1])  # reverse operation: (bs, label_size, K)
        # unnormalized = tf.transpose(tf.reshape(unnormalized, [batch_size, K, self.label_size]),
        #                      perm=[0, 2, 1])  # reverse operation: (bs, label_size, K)
        # for k in range(K):
        #     if k == 0:  # (bs, msg_size) --> (bs, label_size, 1)
        #         p_gnn = tf.expand_dims(self.readout_mlp(tf.squeeze(u[:,:,k])), axis=-1)
        #     else:
        #         p_gnn = tf.concat([p_gnn, tf.expand_dims(self.readout_mlp(tf.squeeze(u[:,:,k])), axis=-1)], axis=2)
        return p_gnn


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


def build_EP(trainSet):  # todo: EP-complex
    T = trainSet.T
    Mr, Nt, mu, SNR = trainSet.Mr, trainSet.Nt, trainSet.mu, trainSet.snr
    lr, maxit = trainSet.lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    prob, test = trainSet.prob, trainSet.test
    layers = []  # layerinfo:(name,xhat_,newvars)

    H = prob.H_  # (bs, 2M, 2N)
    x = prob.x_  # (bs, 2N, 1)
    y = prob.y_  # (bs, 2M, 1)
    sigma2 = prob.sigma2_  # (bs, 1, 1)
    sample_size = prob.sample_size_  # bs

    HT = tf.transpose(H, perm=[0, 2, 1])
    HTH = tf.matmul(HT, H)
    noise_var = sigma2 / 2

    def inv_sigmoid(y):
        x = - np.log(1 / y - 1)
        return x

    # precompute some tensorflow constants
    eps = tf.constant(5e-7, dtype=tf.float32)
    pth = tf.constant(1e-100, dtype=tf.float64)
    Lambda = 1 / (0.5 * tf.ones_like(x, dtype=tf.float32))
    gamma = tf.zeros_like(x, tf.float32)
    beta = inv_sigmoid(0.2)

    for t in range(T):
        # beta = tf.Variable(float(inv_sigmoid(min(0.1*np.exp(iter/1.5), 0.7))), dtype=tf.float32, name='beta_' + str(t))
        # beta = tf.Variable(inv_sigmoid(0.2), dtype=tf.float32,
        #                    name='beta_' + str(t))
        # compute the mean and covariance matrix
        # (bs, 2N, 2N)  lambda:(bs, 2N, 1) -> (bs, 2N, 2N)
        Sigma = tf.linalg.inv(HTH / noise_var + tf.matrix_diag(tf.squeeze(Lambda)))
        Mu = tf.matmul(Sigma, tf.matmul(HT, y) / noise_var + gamma)  # (bs, 2N, 1)

        # compute the extrinsic mean and covariance matrix
        diag = tf.expand_dims(tf.matrix_diag_part(Sigma), -1)  # (bs, 2N, 2N) -> (bs, 2N, 1)
        vab = tf.divide(1, tf.divide(1, diag) - Lambda)  # (bs, 2N, 1)
        vab = tf.maximum(vab, eps)
        uab = vab * (Mu / diag - gamma)  # (bs, 2N, 1)

        # compute the posterior mean and covariance matrix
        vab = tf.cast(vab, dtype=tf.float64)
        uab = tf.cast(uab, dtype=tf.float64)
        ub, vb = nle(mu, uab, vab, pth)
        ub = tf.cast(ub, dtype=tf.float32)
        vb = tf.maximum(tf.cast(vb, dtype=tf.float32), 0.1 * eps)
        vab = tf.cast(vab, dtype=tf.float32)
        uab = tf.cast(uab, dtype=tf.float32)

        # moment matching and damping
        gamma_last = gamma
        Lambda_last = Lambda
        gamma = ub / vb - uab / vab
        Lambda = 1 / vb - 1 / vab
        # gamma = tf.cond(tf.reduce_any(tf.less(Lambda,0)), lambda: gamma_last, lambda:gamma)
        # Lambda = tf.cond(tf.reduce_any(tf.less(Lambda, 0)), lambda: Lambda_last, lambda: Lambda)
        gamma = tf.math.sigmoid(beta) * gamma + (1 - tf.math.sigmoid(beta)) * gamma_last
        Lambda = tf.math.sigmoid(beta) * Lambda + (1 - tf.math.sigmoid(beta)) * Lambda_last

        layers.append(('EP T={0}'.format(t), uab, (beta,)))

    loss_ = tf.nn.l2_loss(uab - x)
    lr_ = tf.Variable(lr, name='lr', trainable=False)
    if tf.trainable_variables() is not None:
        train = tf.train.AdamOptimizer(lr_).minimize(loss_, var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))

    for name, _, var_list in layers:
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])
        done = np.append(done, name)
        print(name + ' ' + describe_var_list)
    print(log)

    if test:
        return sess, uab

    loss_history = []
    save = {}  # for the best model
    ivl = 1

    _, _, _, _, yval, xval, Hval, sigma2val = sample_gen(trainSet, 1, vsample_size)

    for i in range(maxit + 1):
        y, x, H, sigma2, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)

        if i % ivl == 0:  # validation:don't use optimizer
            loss = sess.run(loss_, feed_dict={prob.y_: yval,
                                              prob.x_: xval, prob.H_: Hval, prob.sigma2_: sigma2val,
                                              prob.sample_size_: vsample_size})  # 1000 samples and labels
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
            if i % (100 * ivl) == 0:
                print('')

        for m in range(total_batch):
            sess.run(train, feed_dict={prob.y_: y[m * batch_size:(m + 1) * batch_size],
                                       prob.x_: x[m * batch_size:(m + 1) * batch_size],
                                       prob.H_: H[m * batch_size:(m + 1) * batch_size],
                                       prob.sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                       prob.sample_size_: batch_size})  # 1000 samples and labels
    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin())

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    return sess, uab


# TODO:use py_function and decide which one is better -- results: py_function is too slow
def CG(u, p, residual, r_norm, XI, sample_size):
    # compute the approximate solution based on prior conjugate direction and residual
    XI_p = tf.matmul(XI, p)  # bs*2M*1
    a = r_norm / tf.matmul(tf.transpose(p, perm=[0, 2, 1]), XI_p)
    u = tf.add(u, a * p)
    # compute conjugate direction and residual
    residual = tf.add(residual, -a * XI_p)
    r_norm_last = r_norm
    r_norm = tf.reshape(tf.square(tf.norm(residual, axis=(1, 2))), [sample_size, 1, 1])
    # r_norm_last = tf.maximum(r_norm_last,tf.constant(1e-20))
    b = r_norm / r_norm_last
    p = tf.add(residual, b * p)
    # r_norm = tf.maximum(r_norm, tf.constant(1e-20))
    return u, p, residual, r_norm


def build_CG_OAMP(trainSet):
    T, I = trainSet.T, trainSet.icg
    use_OFDM, K, CP, CP_flag = trainSet.use_OFDM, trainSet.K, trainSet.CP, trainSet.CP_flag
    if use_OFDM:
        Mr, Nt = trainSet.Mr * K, trainSet.Nt * K
    else:
        Mr, Nt = trainSet.Mr, trainSet.Nt
    mu, SNR = trainSet.mu, trainSet.snr
    version = trainSet.version
    lr, maxit = trainSet.lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    prob, test = trainSet.prob, trainSet.test
    layers = []  # layerinfo:(name,xhat_,newvars)

    H_ = prob.H_  # 2M*2N
    x_ = prob.x_  # bs*2N*1
    y_ = prob.y_  # bs*2M*1
    sigma2_ = prob.sigma2_  # bs*1*1
    sample_size = prob.sample_size_  # bs
    eigvalue = prob.eigvalue_  # M*1

    HT_ = tf.transpose(H_, perm=[0, 2, 1])
    HHT = tf.matmul(H_, HT_)
    OneOver_trHTH = tf.reshape(1 / tf.trace(tf.matmul(HT_, H_)), [sample_size, 1, 1])
    sigma2_I = sigma2_ / 2 * tf.eye(2 * Mr, batch_shape=[sample_size], dtype=tf.float32)

    # precompute some tensorflow constants
    epsilon = tf.constant(1e-10, dtype=tf.float32)
    rth = tf.constant(1e-4, dtype=tf.float64)
    pth = tf.constant(1e-100, dtype=tf.float64)
    v_sqr_last = tf.constant(0, dtype=tf.float32)
    x_hat = tf.zeros_like(x_, dtype=tf.float32)

    for t in range(T):
        theta_ = tf.Variable(float(1), dtype=tf.float32, name='theta_' + str(t))
        gamma_ = tf.Variable(float(1), dtype=tf.float32, name='gamma_' + str(t))
        beta_ = tf.Variable(float(0.5), dtype=tf.float32, name='beta_' + str(t))
        if version == 1:
            phi_ = tf.Variable(float(1), dtype=tf.float32, name='phi_' + str(t))
            xi_ = tf.Variable(float(0), dtype=tf.float32, name='xi_' + str(t))

        p_noise = y_ - tf.matmul(H_, x_hat)  # bs*2M*1
        v_sqr = (tf.reshape(tf.square(tf.norm(p_noise, axis=(1, 2))),
                            [sample_size, 1, 1]) - Mr * sigma2_) * OneOver_trHTH  # bs*1*1
        v_sqr = beta_ * v_sqr + (1 - beta_) * v_sqr_last
        v_sqr = tf.maximum(v_sqr, epsilon)
        v_sqr_last = v_sqr

        # with tf.device("/cpu:0"):
        # CG for the calculation of u=(xi)**-1@p_noise
        XI = tf.cast(HHT + sigma2_I / v_sqr, dtype=tf.float64)  # 2M*2M
        # initial value
        u = tf.zeros_like(y_, dtype=tf.float64)
        residual = tf.cast(p_noise, dtype=tf.float64)
        p = residual  # bs*2M*1
        r_norm = tf.reshape(tf.square(tf.norm(residual, axis=(1, 2))), [sample_size, 1, 1])  # bs*1*1
        # build CG into a while_loop
        i_, u_, _, _, r_norm_ = tf.while_loop(
            cond=lambda i, u, p, residual, r_norm: tf.reduce_any(tf.greater(r_norm, rth)),
            body=lambda i, u, p, residual, r_norm: (i + 1, *CG(u, p, residual, r_norm, XI, sample_size)),
            loop_vars=(tf.constant(0), u, p, residual, r_norm), maximum_iterations=I)
        u_ = tf.cast(u_, dtype=tf.float32)

        estimate = tf.reduce_mean(eigvalue / tf.add(eigvalue, sigma2_ / 2 / v_sqr),
                                  1) * 2 * Mr  # add and divide support broadcasting bs*1
        nor_coef = 2 * Nt / tf.reshape(estimate, [sample_size, 1, 1])  # reshape bs*1*1
        r = x_hat + gamma_ * nor_coef * tf.matmul(HT_, u_)
        tau_sqr = v_sqr * ((theta_ ** 2) * nor_coef - 2 * theta_ + 1)  # bs*1*1
        tau_sqr = tf.cast(tf.maximum(tau_sqr, epsilon), dtype=tf.float64)
        r = tf.cast(r, dtype=tf.float64)
        x_hat = nle(mu, r, tau_sqr, pth)

        x_hat = tf.cast(x_hat, dtype=tf.float32)
        r = tf.cast(r, dtype=tf.float32)
        if version == 1:
            x_hat = phi_ * (x_hat - xi_ * r)

        if version == 0:
            layers.append(('CG-OAMP T={0}'.format(t), x_hat, (theta_, gamma_,)))
        else:
            layers.append(('CG-OAMP T={0}'.format(t), x_hat, (theta_, gamma_, beta_, phi_, xi_,)))

    loss_ = tf.nn.l2_loss(x_hat - x_)
    lr_ = tf.Variable(lr, name='lr', trainable=False)
    if tf.trainable_variables() is not None:
        train = tf.train.AdamOptimizer(lr_).minimize(loss_, var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))

    for name, _, var_list in layers:
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + \
                                ','.join([v.name for v in tf.trainable_variables()])
        done = np.append(done, name)
        print(name + ' ' + describe_var_list)
    print(log)

    if test:
        return sess, x_hat

    loss_history = []
    save = {}  # for the best model
    ivl = 1

    if use_OFDM:
        from .MIMO_OFDM_detection import sample_gen_MIMO_OFDM
        yval, xval, Hval, sigma2val, eigval = sample_gen_MIMO_OFDM(trainSet, vsample_size,
                                                                   training_flag=False)
    else:
        _, _, _, _, _, yval, xval, Hval, sigma2val, eigval = sample_gen(trainSet, 1, vsample_size)
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    for i in range(maxit + 1):
        if use_OFDM:
            # for MIMO-OFDM model of all subcarriers
            y, x, H, sigma2, eig = sample_gen_MIMO_OFDM(trainSet, batch_size * total_batch, training_flag=True)
        else:
            y, x, H, sigma2, eig, _, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)
        # TODO:shuffling after every epoch -- not easy -- the order of each elements should match
        if i % ivl == 0:  # validation:don't use optimizer
            loss = sess.run(loss_, feed_dict={prob.y_: yval,
                                              prob.x_: xval, prob.H_: Hval, prob.sigma2_: sigma2val,
                                              prob.sample_size_: vsample_size,
                                              prob.eigvalue_: eigval}, options=run_opts)
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')

            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            # TODO:change back to early stopping
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)
            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'
                             .format(i=i, loss=loss, best=loss_best))
            sys.stdout.flush()
            if i % (100 * ivl) == 0:
                print('')

        for m in range(total_batch):
            sess.run(train, feed_dict={prob.y_: y[m * batch_size:(m + 1) * batch_size],
                                       prob.x_: x[m * batch_size:(m + 1) * batch_size],
                                       prob.H_: H[m * batch_size:(m + 1) * batch_size],
                                       prob.sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                       prob.sample_size_: batch_size,
                                       prob.eigvalue_: eig[m * batch_size:(m + 1) * batch_size]},
                     options=run_opts)

    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin())

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    return sess, x_hat


def build_OAMP(test=False, trainSet=None):
    T = trainSet.T
    use_OFDM, K, CP, CP_flag = trainSet.use_OFDM, trainSet.K, trainSet.CP, trainSet.CP_flag
    if use_OFDM:
        Mr, Nt = trainSet.m * K, trainSet.n * K
    else:
        Mr, Nt = trainSet.m, trainSet.n
    mu, SNR = trainSet.mu, trainSet.snr
    version = 1
    channel_type = trainSet.channel_type
    lr, maxit = trainSet.lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    directory = './model/' + 'OAMPNet_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainSet.prob
    layers = []  # layerinfo:(name,xhat_,newvars)

    H_ = prob.H_
    x_ = prob.x_
    y_ = prob.y_
    sigma2_ = prob.sigma2_
    sample_size = prob.sample_size_

    # precompute some tensorflow constants
    epsilon = tf.constant(1e-10, dtype=tf.float64)
    pth = tf.constant(1e-100, dtype=tf.float64)
    HT_ = tf.transpose(H_, perm=[0, 2, 1])
    HHT = tf.matmul(H_, HT_)
    OneOver_trHTH = tf.reshape(1 / tf.trace(tf.matmul(HT_, H_)), [sample_size, 1, 1])
    sigma2_I = sigma2_ / 2 * tf.eye(2 * Mr, batch_shape=[sample_size], dtype=tf.float64)

    v_sqr_last = tf.constant(0, dtype=tf.float64)
    x_hat = tf.zeros_like(x_, dtype=tf.float64)

    for t in range(T):
        theta_ = tf.Variable(float(1), dtype=tf.float64, name='theta_' + str(t))
        gamma_ = tf.Variable(float(1), dtype=tf.float64, name='gamma_' + str(t))
        if version == 1:
            phi_ = tf.Variable(float(1), dtype=tf.float64, name='phi_' + str(t))
            xi_ = tf.Variable(float(0), dtype=tf.float64, name='xi_' + str(t))
        p_noise = y_ - tf.matmul(H_, x_hat)
        v_sqr = (tf.reshape(tf.square(tf.norm(p_noise, axis=(1, 2))),
                            [sample_size, 1, 1]) - Mr * sigma2_) * OneOver_trHTH
        v_sqr = 0.5 * v_sqr + 0.5 * v_sqr_last
        v_sqr = tf.maximum(v_sqr, epsilon)
        v_sqr_last = v_sqr

        # with tf.device("/cpu:0"):
        w_hat = tf.matmul(HT_, tf.linalg.inv(HHT + sigma2_I / v_sqr))
        nor_coef = 2 * Nt / tf.reshape(tf.trace(tf.matmul(w_hat, H_)), [sample_size, 1, 1])
        r = x_hat + gamma_ * nor_coef * tf.matmul(w_hat, p_noise)

        tau_sqr = v_sqr * ((theta_ ** 2) * nor_coef - 2 * theta_ + 1)
        tau_sqr = tf.cast(tf.maximum(tau_sqr, epsilon), dtype=tf.float64)
        r = tf.cast(r, dtype=tf.float64)
        x_hat, _, _ = nle(mu, r, tau_sqr, pth)

        # x_hat = tf.cast(x_hat, dtype=tf.float32)
        # r = tf.cast(r, dtype=tf.float32)
        if version == 1:
            x_hat = phi_ * (x_hat - xi_ * r)  # (18)

        if version == 0:
            layers.append(('OAMP T={0}'.format(t), x_hat, (theta_, gamma_,)))
        else:
            layers.append(('OAMP T={0}'.format(t), x_hat, (theta_, gamma_, phi_, xi_,)))

    loss_ = tf.nn.l2_loss(x_hat - x_)
    lr_ = tf.Variable(lr, name='lr', trainable=False)
    if tf.trainable_variables() is not None:
        train = tf.train.AdamOptimizer(lr_).minimize(loss_, var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))

    for name, _, var_list in layers:
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])
        done = np.append(done, name)
        print(name + ' ' + describe_var_list)
    print(log)

    if test:
        return sess, x_hat

    if channel_type == '3gpp':
        if Nt == 16:
            Hdataset = sio.loadmat('../MCMC/data/H_3gpp_64x16.mat')['H1']
        elif Nt == 8:
            Hdataset = sio.loadmat('../MCMC/data/H_3gpp_20mhz.mat')['H1']
        trainSet.Hdataset = Hdataset

    loss_history = []
    save = {}  # for the best model
    ivl = 1

    if use_OFDM:
        from .MIMO_OFDM_detection import sample_gen_MIMO_OFDM
        yval, xval, Hval, sigma2val = sample_gen_MIMO_OFDM(trainSet, vsample_size,
                                                           training_flag=False)
    else:
        _, _, _, _, _, _, yval, xval, Hval, sigma2val, labelval, rw_inv_val = sample_gen(trainSet, 1, vsample_size)
    for i in range(maxit + 1):
        if use_OFDM:
            # for MIMO-OFDM model of all subcarriers
            y, x, H, sigma2 = sample_gen_MIMO_OFDM(trainSet, batch_size * total_batch, training_flag=True)
        else:
            y, x, H, sigma2, label, rw_inv, _, _, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)

        if i % ivl == 0:  # validation:don't use optimizer
            loss = sess.run(loss_, feed_dict={prob.y_: yval,
                                              prob.x_: xval, prob.H_: Hval, prob.sigma2_: sigma2val,
                                              prob.sample_size_: vsample_size})  # 1000 samples and labels
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
            if i % (100 * ivl) == 0:
                print('')

        for m in range(total_batch):
            sess.run(train, feed_dict={prob.y_: y[m * batch_size:(m + 1) * batch_size],
                                       prob.x_: x[m * batch_size:(m + 1) * batch_size],
                                       prob.H_: H[m * batch_size:(m + 1) * batch_size],
                                       prob.sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                       prob.sample_size_: batch_size})  # 1000 samples and labels
    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin())

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    para = {}
    for k, v in np.load(savefile).items():
        para[k.replace(':', '').replace('/', '_')] = v
    sio.savemat(savefile.replace('.npz', '.mat'), para)

    return
