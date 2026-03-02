#!/usr/bin/python

import sys
import numpy as np
import numpy.linalg as la
import math
from .utils import QAM_Modulation, NLE, de2bi, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation

beta = np.zeros(20)
para = {}
# loading the trained model parameters
try:
    for k,v in np.load("EP_4×4_16QAM_15dB_I_1.npz").items():
        para[k] = v
except IOError:
    print("no such file")
    pass
# get parameters for CG-OAMP-NET
for t in range(20):
    if para.get("beta_"+str(t)+":0",-1) != -1:
        beta[t] = para["beta_"+str(t)+":0"]
beta = 1. / (1. + np.exp(-beta))


def EP(x,A,y,noise_var,pp_llr,iter,T=10,mu=2):

    # initialize
    beta = min(0.1*np.exp(iter/1.5), 0.7) * np.ones(T)
    M = A.shape[0]
    N = A.shape[1]
    eps = 1e-16
    AT = np.conj(A.T)
    ATA = AT@A
    MSE = np.zeros(T)
    llr_d_data = np.zeros((N, mu))
    # Relation between bin_array and constellation_norm should be the same as MAPPING !!!
    bin_array = np.sign(de2bi(np.arange(2 ** mu), mu) - 0.5).astype(np.int)  # (2 ** mu, mu)
    if mu == 2:
        constellation_norm = _QPSK_Constellation.reshape(-1) / np.sqrt(2)
    elif mu == 4:
        constellation_norm = _16QAM_Constellation.reshape(-1) / np.sqrt(10)
    else:
        constellation_norm = _64QAM_Constellation.reshape(-1) / np.sqrt(42)

    # reverse the LLR definition
    # pp_llr = -pp_llr

    # calculate soft estimates-- mean and  variance of constellation
    dist = 0.5 * bin_array @ pp_llr   # (2**mu, N)
    dist += np.amin(dist, axis=0)
    probs = np.exp(dist).T              # (N, 2**mu)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    s_est = np.sum(probs*constellation_norm, axis=1, keepdims=True)   # (N, 1)
    e_est = np.sum(probs*abs(s_est*np.ones((N, 2**mu)) - constellation_norm) ** 2, axis=1, keepdims=True)  # (N, 1)
    e_est = np.maximum(e_est, 1e-8)

    # calculate the initial pair for EP
    Lambda = 1 / e_est      # (N, 1)
    gamma = s_est * Lambda  # (N, 1)

    for t in range(T):
        # compute the mean and covariance matrix
        Sigma = la.inv(ATA/noise_var+np.diag(Lambda.reshape(N)))
        Mu = Sigma @ (AT@y/noise_var+gamma)

        # compute the extrinsic mean and covariance matrix
        vab = np.real(1/(1/np.diag(Sigma).reshape(N,1)-Lambda))
        uab = vab*(Mu/np.diag(Sigma).reshape(N,1)-gamma)
        MSE[t] = np.mean(abs(x - uab) ** 2)
        # todo: why not use ub as output? —— current answer extrinsic prob is enough like OAMP, but the hard output
        #  should be ub as the diagram show
        # compute the posterior mean and covariance matrix
        ext_probs = np.exp(-abs(uab*np.ones((N, 2**mu)) - constellation_norm)**2 / vab)
        post_probs = probs * ext_probs
        post_probs = post_probs / np.sum(post_probs, axis=1, keepdims=True)  # (N, 2**mu)
        ub = np.sum(post_probs*constellation_norm, axis=1, keepdims=True)
        vb = np.sum(post_probs*abs(ub*np.ones((N, 2**mu)) - constellation_norm) ** 2, axis=1, keepdims=True)
        vb = np.maximum(vb,1e-8)

        # moment matching and damping
        gamma_last = gamma
        Lambda_last = Lambda
        gamma = ub/vb - uab/vab
        Lambda = 1/vb - 1/vab
        idx = Lambda < 0
        Lambda[idx] = Lambda_last[idx]
        gamma[idx] = gamma_last[idx]
        # damping
        gamma = beta[t]*gamma +(1-beta[t]) *gamma_last
        Lambda = beta[t]*Lambda +(1-beta[t]) *Lambda_last

    return ext_probs, MSE

    # # soft output and extrinsic LLR
    # for n in range(N):
    #     for b in range(mu):
    #         pos, neg = 0., 0.
    #         for z in range(2**mu):
    #             if bin_array[z,b] == 1:
    #                 pos += ext_probs[n, z]
    #             else:
    #                 neg += ext_probs[n, z]
    #         llr_d_data[n, b] = np.log(pos+eps)-np.log(neg+eps)
    #
    # # output extrinsic information
    # llr_e_data = llr_d_data
    #
    # return llr_e_data, MSE


def EP_real_v1(x,A,y,noise_var,T=10,mu=2):  # Mu as output

    # initialize
    M = A.shape[0]
    N = A.shape[1]
    gamma = np.zeros((N,1))
    Lambda = 1 / (np.ones(N)/2)
    beta = 0.01
    epsilon = 1e-4
    AT = A.T
    ATA = AT@A
    MSE = np.zeros(T)

    for t in range(T):
        # compute the mean and covariance matrix
        Sigma = la.inv(ATA/noise_var+np.diag(Lambda))
        Mu = Sigma @ (AT@y/noise_var+gamma)
        MSE[t] = np.mean((x-Mu)**2)

        # compute the extrinsic mean and covariance matrix
        vab = 1/(1/np.diag(Sigma).reshape(N,1)-Lambda.reshape(N,1))
        uab = vab*(Mu/np.diag(Sigma).reshape(N,1)-gamma)

        # compute the posterior mean and covariance matrix
        _,_,ub,vb = NLE(vab,uab,orth=False,mu=mu,EP=True,norm=np.sqrt(1))
        vb = np.maximum(vb,5e-7)

        # update gamma and Lambda
        gamma_last = gamma
        Lambda_last = Lambda
        gamma = ub/vb - uab/vab
        Lambda = (1/vb - 1/vab).reshape(N)
        idx = Lambda < 0
        Lambda[idx] = Lambda_last[idx]
        gamma[idx] = gamma_last[idx]
        # damping
        gamma = beta*gamma +(1-beta) *gamma_last
        Lambda = beta*Lambda +(1-beta) *Lambda_last

        v1 = la.norm(gamma - gamma_last) / la.norm(gamma_last)
        v2 = la.norm(Lambda - Lambda_last) / la.norm(Lambda_last)
        if v1 <= epsilon and v2 <= epsilon:
            print(t)
            MSE[t + 1:T] = MSE[t]
            break

    Sigma = la.inv(ATA/noise_var+np.diag(Lambda))
    Mu = Sigma @ (AT@y/noise_var+gamma)
    xhat = Mu

    return xhat,MSE


def EP_real_v2(x,A,y,noise_var,T=10,mu=2):  # ub as output
    # T = T+1
    # initialize
    M = A.shape[0]
    N = A.shape[1]
    gamma = np.zeros((N,1))
    Lambda = 1 / (np.ones(N)/2)
    beta = 0.05
    AT = A.T
    ATA = AT@A
    MSE = np.zeros(T)

    for t in range(T):
        # compute the mean and covariance matrix
        Sigma = la.inv(ATA/noise_var+np.diag(Lambda))
        Mu = Sigma @ (AT@y/noise_var+gamma)
        MSE[t] = np.mean((x-Mu)**2)

        # compute the extrinsic mean and covariance matrix
        vab = 1/(1/np.diag(Sigma).reshape(N,1)-Lambda.reshape(N,1))
        uab = vab*(Mu/np.diag(Sigma).reshape(N,1)-gamma)

        # compute the posterior mean and covariance matrix
        _,_,ub,vb = NLE(vab,uab,orth=False,mu=mu,EP=True,norm=np.sqrt(1))
        vb = np.maximum(vb,5e-7)

        # update gamma and Lambda
        gamma_last = gamma
        Lambda_last = Lambda
        gamma = ub/vb - uab/vab
        Lambda = (1/vb - 1/vab).reshape(N)
        idx = Lambda < 0
        Lambda[idx] = Lambda_last[idx]
        gamma[idx] = gamma_last[idx]
        # damping
        gamma = beta*gamma +(1-beta) *gamma_last
        Lambda = beta*Lambda +(1-beta) *Lambda_last

    return ub,MSE


def EP_real_v3(x,A,y,noise_var,T=10,mu=2,soft=False,pp_llr=None):  # ub as output, stable
    # T = T+1
    # initialize
    M = A.shape[0]
    N = A.shape[1]
    # gamma = np.zeros((N,1))
    # Lambda = 1 / (np.ones(N)/2)
    beta = 0.2
    AT = A.T
    ATA = AT@A
    MSE = np.zeros(T)
    if pp_llr is None:
        pp_llr = np.zeros((mu//2, N))
    else:
        pp_llr = np.concatenate((pp_llr[:,:mu//2], pp_llr[:,mu//2:]), axis=0)
    bin_array = np.sign(de2bi(np.arange(2 ** (mu // 2)), mu//2) - 0.5).astype(np.int32)  # (2 ** mu, mu)
    if mu == 2:  # (0 1) --> (-1 +1)
        constellation_norm = np.array([-1, +1]) / np.sqrt(2)
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +3, +1]) / np.sqrt(10)
    else:
        constellation_norm = np.array([-7, -5, -1, -3, +7, +5, +1, +3]) / np.sqrt(42)

    # calculate soft estimates-- mean and  variance of constellation
    dist = 0.5 * bin_array @ pp_llr  # (2**(mu//2), N)
    dist += np.amin(dist, axis=0)
    probs = np.exp(dist).T  # (N, 2**(mu//2))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    s_est = np.sum(probs * constellation_norm, axis=1, keepdims=True)  # (N, 1)
    e_est = np.sum(probs * (s_est * np.ones((N, 2 ** (mu // 2))) - constellation_norm) ** 2, axis=1,
                   keepdims=True)  # (N, 1)
    e_est = np.maximum(e_est, 1e-8)

    # calculate the initial pair for EP
    Lambda = (1 / e_est).reshape(N)  # (N,)
    gamma = s_est * Lambda.reshape(N,1)  # (N, 1)

    for t in range(T):
        # compute the mean and covariance matrix
        Sigma = la.inv(ATA + noise_var * np.diag(Lambda))
        Mu = Sigma @ (AT@y + noise_var * gamma)
        MSE[t] = np.mean((x-Mu)**2)

        # compute the extrinsic mean and covariance matrix
        diag = noise_var * np.diag(Sigma).reshape(N,1)
        vab = diag/(1 - diag * Lambda.reshape(N,1))
        vab = np.maximum(vab, 5e-7)
        uab = vab*(Mu/diag - gamma)

        # compute the posterior mean and covariance matrix
        if soft:
            _, _, ub, vb, ext_probs = NLE(vab, uab, orth=False, mu=mu, EP=True, norm=np.sqrt(1), soft=True)
            ext_probs = np.maximum(np.exp(-(uab*np.ones((N, 2 ** (mu//2))) - constellation_norm) ** 2 / (2*vab)), 1e-100)
            post_probs = probs * ext_probs
            post_probs = post_probs / np.sum(post_probs, axis=1, keepdims=True)  # (N, 2 ** (mu//2))
            ub = np.sum(post_probs * constellation_norm, axis=1, keepdims=True)
            vb = np.sum(post_probs * (ub * np.ones((N, 2 ** (mu//2))) - constellation_norm) ** 2, axis=1, keepdims=True)
        else:
            _, _, ub, vb = NLE(vab,uab,orth=False,mu=mu,EP=True,norm=np.sqrt(1))
        vb = np.maximum(vb,5e-13)

        # update gamma and Lambda
        gamma_last = gamma
        Lambda_last = Lambda
        gamma = (ub*vab - uab*vb) / vb / vab
        Lambda = ((vab -vb) / vb / vab).reshape(N)
        idx = Lambda < 0
        Lambda[idx] = Lambda_last[idx]
        gamma[idx] = gamma_last[idx]
        # damping
        gamma = beta*gamma +(1-beta) *gamma_last
        Lambda = beta*Lambda +(1-beta) *Lambda_last
        # if t >= 1:
        #     if MSE[t] > MSE[t-1]:
        #         pass
    if soft:
        return ub, MSE, ext_probs
    return ub, MSE


def CG_EP(x, A, y, noise_var, T=10, mu=2):
    # initialize
    N = A.shape[1]
    gamma = np.zeros((N, 1))
    Lambda = np.ones(N) / 2
    beta = 0.2
    epsilon = 1e-4
    AT = A.T
    ATA = AT @ A
    MSE = np.zeros(T)
    iter = np.zeros(T, dtype=int)
    # initialization
    _, s, vt = la.svd(A)
    v = vt.T
    Sigma_0 = np.sum(v / (s**2 / noise_var + 0.5) * v, axis=1).reshape(N, 1)
    u = np.zeros((N, 1))
    xi = ATA / noise_var + np.diag(Lambda)
    residual = AT @ y / noise_var + gamma  # (N, 1)
    Mu, _ = cg(u, residual, xi)

    Sigma_reciprocal = np.diag(ATA / noise_var).reshape(N, 1)

    for t in range(T):
        MSE[t] = np.mean((x - Mu) ** 2)
        if t == 0:
            # compute the extrinsic mean and covariance
            vab = 1 / (1 / Sigma_0 - Lambda.reshape(N, 1))
            uab = vab * (Mu / Sigma_0 - gamma)
        else:
            vab = 1 / Sigma_reciprocal
            uab = vab * (Mu * (Sigma_reciprocal + Lambda.reshape(N, 1)) - gamma)

        # compute the posterior mean and covariance
        _, _, ub, vb = NLE(vab, uab, orth=False, mu=mu, EP=True)
        vb = np.maximum(vb, 5e-7)

        # update gamma and Lambda
        gamma_last = gamma
        Lambda_last = Lambda
        gamma = ub / vb - uab / vab
        Lambda = (1 / vb - 1 / vab).reshape(N)
        idx = Lambda < 0
        Lambda[idx] = Lambda_last[idx]

        # damping
        gamma = beta * gamma + (1 - beta) * gamma_last
        Lambda = beta * Lambda + (1 - beta) * Lambda_last

        # stop criterion
        v1 = la.norm(gamma - gamma_last) / la.norm(gamma_last)
        v2 = la.norm(Lambda - Lambda_last) / la.norm(Lambda_last)
        if v1 <= epsilon and v2 <= epsilon:
            print(t)
            break
        # compute the mean
        xi = ATA / noise_var + np.diag(Lambda)
        residual = AT @ y / noise_var + gamma
        Mu, iter[t] = cg(u, residual, xi)

    xhat = Mu

    return xhat, MSE


def cg(u, residual, xi, icg = 50):
    p = residual
    r_norm = residual.T @ residual
    for i in range(icg):
        # compute the approximate solution based on prior conjugate direction and residual
        xi_p = xi @ p #bs*2M*1
        a = r_norm / (p.T @ xi_p)
        u += a*p
        # compute conjugate direction and residual
        residual -= a * xi_p
        r_norm_last = r_norm
        r_norm = residual.T @ residual
        b = r_norm / r_norm_last
        p = residual + b*p
        if r_norm < 1e-4:
            # print(i)
            break

    return u, i


def EPA(x, A, y, noise_var, T=10, mu=2):
    """
    moment matching avoid mu, gamma, lambda
    Args:
        x ():
        A ():
        y ():
        noise_var ():
        T ():
        mu ():

    Returns:

    """
    # initialize
    N = A.shape[1]
    Lambda = np.ones(N) / 2
    rho = np.zeros((N, 1))
    beta = 0.2
    epsilon = 1e-4
    AT = A.T
    ATA = AT @ A
    MSE = np.zeros(T)
    # initialization
    _, s, vt = la.svd(A)
    v = vt.T
    Sigma_0 = np.sum(v / (s ** 2 / noise_var + 0.5) * v, axis=1).reshape(N, 1)
    u = np.zeros((N, 1))
    xi = ATA / noise_var + np.diag(Lambda)
    residual = AT @ y / noise_var  # (N, 1)
    Mu, _ = cg(u, residual, xi)
    Sigma_reciprocal = np.diag(ATA / noise_var).reshape(N, 1)
    for t in range(T):
        if t == 0:
            # compute the extrinsic mean and covariance
            vab = 1 / (1 / Sigma_0 - Lambda.reshape(N, 1))
            uab = vab * Mu / Sigma_0
        else:
            vab = 1 / Sigma_reciprocal  # h2
            uab = rho / Sigma_reciprocal  # t

        # compute the posterior mean and covariance
        _, _, ub, _ = NLE(vab, uab, orth=False, mu=mu, EP=True)
        m = y - A @ ub
        rho_last = rho
        rho = AT @ m / noise_var + 1 / vab * ub
        # damping
        rho = beta * rho + (1 - beta) * rho_last

        MSE[t] = np.mean((x - ub) ** 2)

    xhat = ub
    return xhat, MSE


def EP_U_SE(A, s, noise_var, T=5, mu=2):
    MSE = np.zeros(T)
    M, N = A.shape
    v_sqr = 0.5
    eigvalue = s ** 2
    simu_times = 50000
    bits = np.random.binomial(n=1, p=0.5, size=(simu_times // 2 * mu, 1))
    x = QAM_Modulation(bits, mu).reshape(simu_times // 2, 1)
    x = np.concatenate((np.real(x), np.imag(x)))  # (simu_times, 1)
    for t in range(T):
        # LE
        estimate = sum(eigvalue / (eigvalue + (noise_var / 2) / v_sqr))  # todo: asymptotic eigenvalue distribution of HTH
        nor_coef = N / estimate
        tau_sqr = v_sqr * (nor_coef - 1)
        tau_sqr = max(tau_sqr, 1e-10)

        # NLE
        n = np.sqrt(tau_sqr) * np.random.randn(simu_times, 1)
        r = x + n
        _, vhat, _, v_sqr = NLE(tau_sqr, r, SE=True, x=x, mu=mu, orth=True)
        MSE[t] = 10 * np.log10(vhat)  # the final MSE is vhat, not v_sqr

    return MSE