"""
Exercise 2.7: Data-Driven SISO-OFDM Channel Estimation

This script contains the `build_ce_dnn` function, which defines
and trains the DNN-based channel estimator using TensorFlow.

TODO:
Complete the `build_ce_dnn` function. You need to define the input/output
placeholders and realize the network architecture and loss function.
"""

import os
import numpy as np
import numpy.linalg as la
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tools.shrinkage as shrinkage
from .train import load_trainable_vars, save_trainable_vars
from .raputil import sample_gen
from tensorflow.keras.layers import Dense


def build_ce_dnn(
        K, 
        SNR, 
        savefile, 
        learning_rate=1e-3, 
        training_epochs=2000, 
        batch_size=50,
        nh1=500, 
        nh2=250, 
        test_flag=False, 
        cp_flag=True, 
        no_cp=False, 
        data_mu=6
    ):
    n_input = 2 * K       # LS estimate as input
    n_output = 2 * K

    os.makedirs(os.path.dirname(savefile), exist_ok=True)

    # please fill in the blank in the following codes
    nn_input = tf.placeholder(tf.float32, [None, n_input], name='nn_input')
    H_true = tf.placeholder(tf.float32, [None, n_output], name='H_true')    # label

    dense1 = Dense(nh1, activation='relu', name='dense1')
    dense2 = Dense(nh2, activation='relu', name='dense2')
    output_layer = Dense(n_output, activation=None, name='output_layer')

    tmp = dense1(nn_input)
    tmp = dense2(tmp)
    H_out = output_layer(tmp)

    # Define loss and optimizer, minimize the l2 loss
    loss_ = tf.reduce_mean(tf.reduce_sum(tf.square(H_out - H_true), axis=1))
    global_step = tf.Variable(0, trainable=False)
    decay_steps, lr_decay = 20000, 0.1
    lr_ = tf.train.exponential_decay(learning_rate, global_step, decay_steps, lr_decay, name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_).minimize(
        loss_,
        global_step,
        var_list=tf.trainable_variables()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    log = str(state.get('log', ''))
    print(log)

    if test_flag:
        if not os.path.isfile(savefile):
            raise FileNotFoundError(
                'Checkpoint not found: {}\nPlease train the DNN first.'.format(savefile)
            )
        return sess, nn_input, H_out

    test_step = 5
    loss_history = []
    save = {}  # for the best model

    val_ls, val_labels, _, _ = sample_gen(
        5000, SNR, training_flag=False, NoCP=no_cp, CP_flag=cp_flag, data_mu=data_mu
    )
    val_sample = val_ls.astype(np.float32)
    val_labels = val_labels.astype(np.float32)

    for epoch in range(training_epochs + 1):
        train_loss = 0.
        for m in range(20):
            batch_ls, batch_labels, _, _ = sample_gen(
                batch_size, SNR, training_flag=True, NoCP=no_cp, CP_flag=cp_flag, data_mu=data_mu
            )
            sample = batch_ls.astype(np.float32)
            batch_labels = batch_labels.astype(np.float32)

            _, loss = sess.run(
                [optimizer, loss_],
                feed_dict={nn_input: sample, H_true: batch_labels}
            )
            train_loss += loss

        sys.stdout.write('\repoch={epoch:<6d} loss={loss:.9f} on train set'.format(epoch=epoch, loss=train_loss))
        sys.stdout.flush()

        # validation
        if epoch % test_step == 0:
            loss = sess.run(loss_, feed_dict={nn_input: val_sample, H_true: val_labels})
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()

            # for the best model
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)

            print("\nepoch={epoch:<6d} loss={loss:.9f} (best={best:.9f}) on test set".format(
                epoch=epoch, loss=loss, best=loss_best))

    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k)

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} iterations'.format(
        loss=loss,
        i=epoch,
        best=loss_best,
        j=loss_history.argmin() * test_step
    )

    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    print("optimization finished")

    return sess, nn_input, H_out