# -*- coding:utf-8 -*-
# @Time  : 2022/9/10 22:39
# @Author: STARain
# @File  : AMP_NN_TF2.py

from __future__ import division
from __future__ import print_function
from .train_tf2 import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir
import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


def train_llr_correction_net(test=False, trainset=None):
    mr, nt, mu, SNR = trainset.m, trainset.n, trainset.mu, trainset.snr
    snr_range = trainset.snr_range
    lr, lr_decay, decay_steps, min_lr, maxit = trainset.lr, trainset.lr_decay, trainset.decay_steps, trainset.min_lr, trainset.maxit
    vsample_size = trainset.vsample_size
    total_batch, batch_size = trainset.total_batch, trainset.batch_size
    tsample_size = total_batch * batch_size
    channel_type, rho_tx, rho_rx = trainset.channel_type, trainset.rho_tx, trainset.rho_rx
    savefile = trainset.savefile
    net = trainset.net
    directory = './model/LC_' + net + '_' + str(mr) + 'x' + str(nt) + '_' + str(2 ** mu) + 'QAM' + \
                '_list' + str(trainset.samples * trainset.samplers)
    mkdir(directory)
    savefile = directory + '/' + savefile
    model = LLRCorrectionNet(trainset=trainset)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, lr_decay, name='lr')
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    @tf.function
    def grad(label, singular, sigma, raw_llr, list_size, bs):
        with tf.GradientTape() as tape:
            loss, corrected_llr = model(label, singular, sigma, raw_llr, list_size, bs)
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_variables), trainset.grad_clip)
        return loss, grads

    state = load_trainable_vars(model, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)
    if test:
        return model

    loss_history = []
    save = {}  # for the best model
    ivl = 5

    dataset = './dataset/Valid_' + net + '_' + str(mr) + 'x' + str(nt) + \
              '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB_list' + str(trainset.samples * trainset.samplers) + \
              (('_' + channel_type + str(rho_tx).replace('.', '')) if channel_type == 'corr' else '') + '.mat'
    valid_dataset, train_dataset = load_dataset(dataset, snr_range)

    val_batch_size = vsample_size // total_batch
    train_dataset_size = train_dataset[0].shape[0]
    train_epoch = train_dataset_size // tsample_size

    for i in range(maxit + 1):
        if i % ivl == 0:  # validation:don't use optimizer
            loss = 0.
            for m in range(total_batch):
                raw_llrbatch = valid_dataset[0][m * val_batch_size: (m + 1) * val_batch_size, :]
                true_llrbatch = valid_dataset[1][m * val_batch_size: (m + 1) * val_batch_size, :]
                sigmabatch = valid_dataset[2][m * val_batch_size: (m + 1) * val_batch_size, :]
                singbatch = valid_dataset[3][m * val_batch_size: (m + 1) * val_batch_size, :]
                loss_batch, corrected_llr_batch = model(true_llrbatch, singbatch, sigmabatch, raw_llrbatch,
                                                        trainset.samples * trainset.samplers, bs=val_batch_size)
                loss += loss_batch
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            if loss == loss_best:
                for v in model.trainable_variables:
                    save[str(v.name)] = v.numpy()

            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'
                             .format(i=i, loss=loss, best=loss_best))
            sys.stdout.flush()
            if i % (100 * 1) == 0:
                print('')

        # generate training dataset
        idx = i % train_epoch
        raw_llr = train_dataset[0][idx * tsample_size: (idx + 1) * tsample_size, :]
        true_llr = train_dataset[1][idx * tsample_size: (idx + 1) * tsample_size, :]
        sigma = train_dataset[2][idx * tsample_size: (idx + 1) * tsample_size, :]
        sing = train_dataset[3][idx * tsample_size: (idx + 1) * tsample_size, :]

        for m in range(total_batch):
            raw_llrbatch = raw_llr[m * batch_size: (m + 1) * batch_size, :]
            true_llrbatch = true_llr[m * batch_size: (m + 1) * batch_size, :]
            sigmabatch = sigma[m * batch_size: (m + 1) * batch_size, :]
            singbatch = sing[m * batch_size: (m + 1) * batch_size, :]
            loss_batch, grads = grad(true_llrbatch, singbatch, sigmabatch, raw_llrbatch,
                                     trainset.samples * trainset.samplers, bs=batch_size)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in model.trainable_variables])
    for k, d in save.items():
        if k in tv:
            tv[k].assign(tf.convert_to_tensor(d))
            print('restoring ' + k)

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin() * ivl)

    state['done'] = done
    state['log'] = log
    save_trainable_vars(model, savefile, **state)

    para = {}
    for k, v in np.load(savefile).items():
        para[k.replace(':', '').replace('/', '_')] = v
    para['loss_history'] = loss_history
    sio.savemat(savefile.replace('.npz', '.mat'), para)

    return


class LLRCorrectionNet(Model):
    def __init__(self, trainset=None):
        super(LLRCorrectionNet, self).__init__()
        self.mr, self.nt, self.mu = trainset.m, trainset.n, trainset.mu
        self.dnn = DNN(output_size=self.nt * self.mu)

    def __call__(self, label, singular, sigma, raw_llr, list_size, bs):
        input_vec = tf.concat([raw_llr / 100, singular, sigma, list_size *
                               tf.ones([bs, 1], dtype=tf.float64)], axis=1)  # (bs, nt*mu + nt + 2)
        corrected_llr = self.dnn(input_vec)  # todo: learn residual
        corrected_llr = tf.minimum(tf.maximum(corrected_llr + raw_llr, -30.0), 30.0)
        label = tf.minimum(tf.maximum(label, -30.0), 30.0)
        est_soft_bits = 1 / (1 + tf.exp(-corrected_llr))
        soft_bits = 1 / (1 + tf.exp(-label))
        loss = - tf.reduce_sum(est_soft_bits * tf.math.log(soft_bits) + (1 - est_soft_bits) * tf.math.log(1 - soft_bits)) \
                       / (tf.cast(bs, dtype=tf.float64) * self.nt * self.mu)
        return loss, corrected_llr


class DNN(tf.Module):
    def __init__(self, n_hidden_1=500, n_hidden_2=250, output_size=8, activation=None):
        super(DNN, self).__init__()
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.dense1 = Dense(n_hidden_1, activation='relu')
        self.dense2 = Dense(n_hidden_2, activation='relu')
        self.output_layer = Dense(output_size, activation=activation)

    def __call__(self, inputs):
        tmp = self.dense1(inputs)
        tmp = self.dense2(tmp)
        return self.output_layer(tmp)


def load_dataset(filename, snr_range):
    raw_llr_validset, true_llr_validset, sigma_validset, singular_validset = [], [], [], []
    raw_llr_trainset, true_llr_trainset, sigma_trainset, singular_trainset = [], [], [], []

    idx_sta, idx_end = filename.find('QAM_') + 4, filename.find('dB')
    for snr in snr_range:
        filename_snr = filename[:idx_sta] + str(snr) + filename[idx_end:]
        dataset = sio.loadmat(filename_snr)
        raw_llr, true_llr, sigma, singular = dataset['raw_llr'], dataset['true_llr'], dataset['sigma'], dataset[
            'singular']
        raw_llr_validset.append(raw_llr)
        true_llr_validset.append(true_llr)
        sigma_validset.append(sigma * np.ones((raw_llr.shape[0], 1)))
        singular_validset.append(singular)

        filename_snr = filename_snr.replace('Valid', 'Train')
        dataset = sio.loadmat(filename_snr)
        raw_llr, true_llr, sigma, singular = dataset['raw_llr'], dataset['true_llr'], dataset['sigma'], dataset[
            'singular']
        raw_llr_trainset.append(raw_llr)
        true_llr_trainset.append(true_llr)
        sigma_trainset.append(sigma * np.ones((raw_llr.shape[0], 1)))
        singular_trainset.append(singular)
    raw_llr_validset, true_llr_validset = np.concatenate(tuple(raw_llr_validset)), np.concatenate(
        tuple(true_llr_validset))
    sigma_validset, singular_validset = np.concatenate(tuple(sigma_validset)), np.concatenate(
        tuple(singular_validset))
    raw_llr_trainset, true_llr_trainset = np.concatenate(tuple(raw_llr_trainset)), np.concatenate(
        tuple(true_llr_trainset))
    sigma_trainset, singular_trainset = np.concatenate(tuple(sigma_trainset)), np.concatenate(
        tuple(singular_trainset))

    return (raw_llr_validset, true_llr_validset, sigma_validset, singular_validset), \
           (raw_llr_trainset, true_llr_trainset, sigma_trainset, singular_trainset)