#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
import numpy as np
import sys
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import tensorflow as tf

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


def build_EP(trainSet):
    
    # ─── YOUR CODE HERE ──────────────────────────────────────────────────── #
    #
    #
    #
    # ─────────────────────────────────────────────────────────────────────── #

    return sess, uab
