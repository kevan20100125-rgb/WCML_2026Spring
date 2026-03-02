# -*- coding:utf-8 -*-
# @Time  : 2022/6/13 22:44
# @Author: STARain
# @File  : mcmc.py
# Implementation of [datta, 2013]

import numpy as np
import numpy.linalg as la


def gibbs_sampling(x, A, y, noise_var, alpha=np.sqrt(2), mu=2, iter=320, randomized=False):
    # initialize
    mr, nt = A.shape
    AT = A.T
    x_hat = la.inv(AT @ A) @ AT @ y  # zero-forcing as the initial solution
    mse = np.zeros(iter + 1)
    mse[0] = np.mean((x - x_hat)**2)
    beta = 100000  # la.norm(y - A @ x_hat, 2, axis=0)
    beta_history = np.zeros(iter + 1)
    beta_history[0] = beta
    c1 = 10 * mu
    if mu == 2:
        constellation_norm = np.array([+1, -1]) / np.sqrt(2)
    elif mu == 4:
        constellation_norm = np.array([+3, +1, -1, -3]) / np.sqrt(10)
    else:
        constellation_norm = np.array([+7, +5, +3, +1, -1, -3, -5, -7]) / np.sqrt(42)
    cardinality = len(constellation_norm)
    # for randomized MCMC: qi = 1 / nt
    if randomized:
        alpha, iter = 1, 8 * nt

    x_hat_i = x_hat.copy()
    for i in range(iter):
        for n in range(nt):
            if randomized:
                k = np.random.randint(low=0, high=nt)
                if n == k:
                    condition_prob = 1 / cardinality * np.ones(cardinality)
                    x_hat_i[n] = np.random.choice(a=constellation_norm, p=condition_prob)  # generates a random sample
                    continue
            scaled_ml_cost = np.zeros(cardinality)
            for c in range(cardinality):
                x_hat_n = x_hat_i.copy()
                x_hat_n[n] = constellation_norm[c]
                scaled_ml_cost[c] = - la.norm(y - A @ x_hat_n, 2, axis=0) ** 2 / (2 * alpha**2 * noise_var)
            condition_prob = np.maximum(np.exp(scaled_ml_cost), 1e-100)  # shape:(|A|,) todo: maximum only for binary
            condition_prob = condition_prob / np.sum(condition_prob)  # todo: max-log approximation
            x_hat_i[n] = np.random.choice(a=constellation_norm, p=condition_prob)  # generates a random sample

        ml_cost = la.norm(y - A @ x_hat_i, 2, axis=0)
        if ml_cost < beta:
            x_hat, beta = x_hat_i.copy(), ml_cost
        mse[i + 1] = np.mean((x - x_hat) ** 2)
        beta_history[i + 1] = beta  # for calculating stalling function
        if randomized:
            quality_metric = np.minimum((la.norm(y - A @ x_hat, 2, axis=0) ** 2 - mr * noise_var) \
                             / (np.sqrt(mr / 2) * 2 * noise_var), 200)
            stall_count = int(np.ceil(max(10, c1 * np.exp(quality_metric))))
            if stall_count < i:
                if beta == beta_history[i + 1 - stall_count]:
                    break

    return x_hat, mse


def gibbs_sampling_para(x, A, y, noise_var, alpha=np.sqrt(2), mu=2, iter=320, randomized=False, samplers=50):
    # initialize
    mr, nt = A.shape
    AT = A.T
    beta = 100000 * np.ones((samplers, 1))  # la.norm(y - A @ x_hat, 2, axis=0)
    beta_history = np.zeros((samplers, iter + 1))
    beta_history[:, 0:1] = beta
    c1 = 10 * mu
    if mu == 2:
        constellation_norm = np.array([+1, -1]) / np.sqrt(2)
    elif mu == 4:
        constellation_norm = np.array([+3, +1, -1, -3]) / np.sqrt(10)
    else:
        constellation_norm = np.array([+7, +5, +3, +1, -1, -3, -5, -7]) / np.sqrt(42)
    cardinality = len(constellation_norm)
    x_hat = constellation_norm[np.random.randint(low=0, high=2 ** (mu // 2), size=(samplers, nt, 1))].copy()
    mse = np.zeros(iter + 1)
    mse[0] = np.mean((x - x_hat) ** 2)
    # for randomized MCMC: qi = 1 / nt
    if randomized:
        alpha = np.sqrt(10)
        # iter = 8 * nt * 2 ** (mu // 2 - 1)

    x_hat_i = x_hat.copy()  # (np, nt, 1)
    for i in range(iter):
        for n in range(nt):
            if randomized:
                k = np.random.randint(low=0, high=nt)
                if n == k:
                    condition_prob = 1 / cardinality * np.ones(cardinality)
                    x_hat_i[:, n] = np.random.choice(a=constellation_norm, p=condition_prob, size=(samplers, 1))  # generates a random sample
                    continue
            scaled_ml_cost = np.zeros((samplers, cardinality))
            for c in range(cardinality):
                x_hat_n = x_hat_i.copy()
                x_hat_n[:, n] = constellation_norm[c] * np.ones((samplers, 1))
                scaled_ml_cost[:, c:c + 1] = - la.norm(y - A @ x_hat_n, 2, axis=1) ** 2 / (2 * alpha**2 * noise_var)
            condition_prob = np.maximum(np.exp(scaled_ml_cost), 1e-100)  # shape:(np, |A|)
            condition_prob = condition_prob / np.sum(condition_prob, axis=1, keepdims=True)
            for p in range(samplers):
                x_hat_i[p, n] = np.random.choice(a=constellation_norm, p=condition_prob[p])  # generates a random sample

        ml_cost = la.norm(y - A @ x_hat_i, 2, axis=1)  # (np, 1)
        idx = np.squeeze(ml_cost < beta)
        if idx.any():
            x_hat[idx], beta[idx] = x_hat_i[idx], ml_cost[idx]
        mse[i + 1] = np.mean((x - x_hat) ** 2)
        beta_history[:, i + 1:i + 2] = beta  # for calculating stalling function
        # if randomized:  # break when all satisfies
        #     quality_metric = np.minimum((la.norm(y - A @ x_hat, 2, axis=1) ** 2 - mr * noise_var) \
        #                      / (np.sqrt(mr / 2) * 2 * noise_var), 200)  # (np, 1)
        #     stall_count = int(np.ceil(max(10, c1 * np.exp(quality_metric))))
        #     if (stall_count < i):
        #         if beta == beta_history[i + 1 - stall_count]:
        #             break

    x_hat = x_hat[np.argmin(beta)]
    return x_hat, mse