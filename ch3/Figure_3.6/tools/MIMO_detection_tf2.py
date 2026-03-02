#!/usr/bin/python
from __future__ import division
import numpy as np
import os
import time
import sys
import math
import numpy.linalg as la
from scipy.linalg import toeplitz, sqrtm, cholesky, dft
from .utils import QAM_Modulation, QAM_Demodulation, indicator, lmmse_ce
from .OAMP import OAMP
from .CG_OAMP import CG_OAMP
from .MAMP import MAMP
from .EP import EP, EP_real_v1, EP_real_v2, EP_real_v3, EP_U_SE
from .mcmc import gibbs_sampling
from .swamp import amp
from .MHGD import mhgd, mhgd_para
from .langevin_numpy import LangevinNumpy, langevin

from gurobipy import *

pi = math.pi


def MIMO_detection_simulate(model, sysin, SNR=40):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    csi = sysin.csi
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    ser = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if csi == 2:
        Np = Nt
        wlmmse, xp = channel_est(sysin, SNR)

    if sysin.detect_type == 'langevin':
        # create noise vector for Langevin
        sigma_gaussian = np.exp(np.linspace(np.log(sysin.sigma_0), np.log(sysin.sigma_L),
                                            sysin.n_sigma_gaussian))

        # create model
        sysin.lang = LangevinNumpy(sigma_gaussian, sysin.n_sample_init, sysin.step_size, sysin.mu)

    norm = np.sqrt(1)
    # num_trail = 10000
    # for i in range(num_trail):
    while True:
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1) / norm

        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx

        # channel input & output
        y = H @ x

        # add AWGN noise
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-(SNR) / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        if csi == 2:
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, Np) + 1j * np.random.randn(Mr, Np))
            yp = H @ xp + noise
            yp_vec = yp.reshape(-1, 1)
            h_hat = wlmmse @ yp_vec
            H = h_hat.reshape(Mr, Nt)

        if sysin.detect_type != 'MHGD' and sysin.detect_type != 'MHGD_PARA':
            # convert complex into real
            x = np.concatenate((np.real(x), np.imag(x)))
            H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                                np.concatenate((np.imag(H), np.real(H)), axis=1)))
            y = np.concatenate((np.real(y), np.imag(y)))

        sta = time.time()
        # if count == 10877:
        #     print('')
        x_hat, MSE = detector(sysin, H, x, y, sigma2, MSE, model)
        end = time.time()

        if sysin.detect_type != 'MHGD' and sysin.detect_type != 'MHGD_PARA':
            # back into np.complex64
            x_hat = x_hat.reshape((2, Nt))
            x_hat = x_hat[0, :] + 1j * x_hat[1, :]

        # Demodulate
        x_hat_demod = QAM_Demodulation(x_hat * norm, mu)

        total_time += (end - sta)

        # calculate BER
        err_bits = np.sum(np.not_equal(x_hat_demod, bits))
        total_err_bits += err_bits
        total_bits += mu * Nt
        count = count + 1
        if err_bits > 0:
            ser += calc_ser(x_hat_demod, bits, Nt, mu)
            sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f} SER={SER:.6f}'
                             .format(teb=total_err_bits, tb=total_bits, BER=total_err_bits / total_bits,
                                     SER=ser / count / Nt))
            sys.stdout.flush()
        if total_err_bits > err_bits_target or total_bits > 1e7:
            end = time.time()
            iter_time = end - start
            print("\nSNR=", SNR, "iter_time:", iter_time)
            ber = total_err_bits / total_bits
            ser = ser / count / Nt
            print("BER:", ber)
            print("SER:", ser)
            print("MSE:", 10 * np.log10(MSE / count))
            break

    # print("time:",total_time/1000)
    return ber, ser, np.array([total_err_bits, total_bits])


def MIMO_detection_batch(model, sysin, SNR=40, batch_size=64):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    csi = sysin.csi
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    ser = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if csi == 2:
        Np = Nt
        wlmmse, xp = channel_est(sysin, SNR)
    norm = np.sqrt(1)
    net_type = {'MHGDNet', 'FSNet', 'CGNet', 'AMPNet', 'AMPNet_MHGD', 'MHCGNet', 'DMHGDNet'}
    if sysin.detect_type not in net_type:
        H_batch = np.zeros((batch_size, 2 * Mr, 2 * Nt))
        x_batch = np.zeros((batch_size, 2 * Nt, 1))
        y_batch = np.zeros((batch_size, 2 * Mr, 1))
        bits_batch = np.zeros((batch_size, Nt * mu), dtype=int)
    else:
        H_batch = np.zeros((batch_size, Mr, Nt), dtype=complex)
        x_batch = np.zeros((batch_size, Nt, 1), dtype=complex)
        y_batch = np.zeros((batch_size, Mr, 1), dtype=complex)
        bits_batch = np.zeros((batch_size, Nt * mu), dtype=int)
    # num_trail = 10000
    # for i in range(num_trail):
    while True:
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1) / norm

        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx

        # channel input & output
        y = H @ x

        # add AWGN noise
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-(SNR) / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        if csi == 2:  # channel estimation
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, Np) + 1j * np.random.randn(Mr, Np))
            yp = H @ xp + noise
            yp_vec = yp.reshape(-1, 1)
            h_hat = wlmmse @ yp_vec
            H = h_hat.reshape(Mr, Nt)

        if sysin.detect_type not in net_type:
            # convert complex into real
            x = np.concatenate((np.real(x), np.imag(x)))
            H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                                np.concatenate((np.imag(H), np.real(H)), axis=1)))
            y = np.concatenate((np.real(y), np.imag(y)))

        # stack
        H_batch[count % batch_size] = H
        x_batch[count % batch_size] = x
        y_batch[count % batch_size] = y
        bits_batch[count % batch_size] = bits

        count = count + 1
        if count % batch_size == 0:
            sta = time.time()
            if sysin.detect_type == 'AMPNet':
                x_hat_batch, _, _ = model(x_batch, y_batch, H_batch,
                                          sigma2 * np.ones((batch_size, 1, 1)),
                                          label=np.zeros((batch_size, 2 ** (mu // 2), 2 * Nt)),
                                          bs=batch_size, test=True)
            elif sysin.detect_type == 'AMPNet_MHGD':
                x_amp_batch, _, _ = model[0](x_batch, y_batch, H_batch,
                                          sigma2 * np.ones((batch_size, 1, 1)),
                                          label=np.zeros((batch_size, 2 ** (mu // 2), 2 * Nt)),
                                          bs=batch_size, test=True)
                x_hat_batch, _, _ = model[1](x_batch, y_batch, H_batch,
                                          sigma2 * np.ones((batch_size, 1, 1)), batch_size, test=True, x_fs=x_amp_batch)
            else:
                x_hat_batch, _, _ = model(x_batch, y_batch, H_batch,
                                          sigma2 * np.ones((batch_size, 1, 1)), batch_size, test=True)
            end = time.time()

            x_hat_batch = x_hat_batch.numpy()
            mse = np.mean(abs(x_hat_batch - x_batch) ** 2)
            MSE += np.array([mse])


            for m in range(batch_size):
                x_hat = x_hat_batch[m]
                if sysin.detect_type not in net_type:
                    # back into np.complex64
                    x_hat = x_hat.reshape((2, Nt))
                    x_hat = x_hat[0, :] + 1j * x_hat[1, :]

                # Demodulate
                x_hat_demod = QAM_Demodulation(x_hat * norm, mu)

                total_time += (end - sta)

                # calculate BER
                err_bits = np.sum(np.not_equal(x_hat_demod, bits_batch[m]))
                total_err_bits += err_bits
                total_bits += mu * Nt

                if err_bits > 0:
                    ser += calc_ser(x_hat_demod, bits_batch[m], Nt, mu)
                    sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f} SER={SER:.6f}'
                                     .format(teb=total_err_bits, tb=total_bits, BER=total_err_bits / total_bits,
                                             SER=ser / (count - batch_size + m + 1) / Nt))
                    sys.stdout.flush()

            if total_err_bits > err_bits_target or total_bits > 1e7:
                end = time.time()
                iter_time = end - start
                print("\nSNR=", SNR, "iter_time:", iter_time)
                ber = total_err_bits / total_bits
                ser = ser / count / Nt
                print("BER:", ber)
                print("SER:", ser)
                print("MSE:", 10 * np.log10(MSE / count * batch_size))
                break

    return ber, ser, np.array([total_err_bits, total_bits])


def calc_ser(x_hat_demod, bits, Nt, mu):
    # ser = 0
    # for n in range(Nt):
    #     err_bits = np.sum(np.not_equal(x_hat_demod[n*mu:(n+1)*mu], bits[n*mu:(n+1)*mu]))
    #     if err_bits > 0:
    #         ser += 1
    err = np.not_equal(x_hat_demod, bits).reshape(Nt, mu)
    ser = np.sum(np.any(err, axis=1))
    return ser


def sample_gen(trainSet, ts, vs, training_flag=True):
    Mr, Nt = trainSet.m, trainSet.n
    mu, SNR = trainSet.mu, trainSet.snr
    channel_type, rho_tx, rho_rx = trainSet.channel_type, trainSet.rho_tx, trainSet.rho_rx
    if training_flag is False:
        ts = 0
    # generate training samples:
    H_ = np.zeros((ts * Mr, Nt), dtype=complex)
    x_ = np.zeros((ts * Nt, 1), dtype=complex)
    y_ = np.zeros((ts * Mr, 1), dtype=complex)
    sigma2_ = np.zeros((ts, 1))
    # generate development samples:
    Hval_ = np.zeros((vs * Mr, Nt), dtype=complex)
    xval_ = np.zeros((vs * Nt, 1), dtype=complex)
    yval_ = np.zeros((vs * Mr, 1), dtype=complex)
    sigma2val_ = np.zeros((vs, 1))

    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if channel_type == 'nr':
        rspat = nr_corr_channel(Mr, Nt, 'meda', a=0.0)
        rherm = cholesky(rspat).T

    for i in range(ts + vs):
        if trainSet.snr == 'varying_':
            SNR = np.random.randint(low=9, high=16)
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1)
        # Rayleigh MIMO channel
        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt) +
                                   1j * np.random.randn(Mr, Nt))
        if channel_type == 'corr':  # correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx
            # H = H * np.sqrt(Nt) / la.norm(H, 'fro')
        if channel_type == 'nr':
            H = np.reshape(rherm @ H.reshape(-1, 1, order='F'), (Mr, Nt), order='F')
        # channel input & output
        y = H @ x
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-SNR / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        # stack
        if i < ts:
            H_[Mr * i:Mr * (i + 1)] = H
            x_[Nt * i:Nt * (i + 1)] = x
            y_[Mr * i:Mr * (i + 1)] = y
            sigma2_[i] = sigma2
        else:
            Hval_[Mr * (i - ts):Mr * (i - ts + 1)] = H
            xval_[Nt * (i - ts):Nt * (i - ts + 1)] = x
            yval_[Mr * (i - ts):Mr * (i - ts + 1)] = y
            sigma2val_[i - ts] = sigma2
    # reshape
    H_ = H_.reshape(ts, Mr, Nt)
    x_ = x_.reshape(ts, Nt, 1)
    y_ = y_.reshape(ts, Mr, 1)
    sigma2_ = sigma2_.reshape(ts, 1, 1)
    Hval_ = Hval_.reshape(vs, Mr, Nt)
    xval_ = xval_.reshape(vs, Nt, 1)
    yval_ = yval_.reshape(vs, Mr, 1)
    sigma2val_ = sigma2val_.reshape(vs, 1, 1)

    return y_, x_, H_, sigma2_, yval_, xval_, Hval_, sigma2val_


def corr_channel(Mr, Nt, rho_tx=0.5, rho_rx=0.5):
    Rtx_vec = np.ones(Nt)
    for i in range(1, Nt):
        Rtx_vec[i] = rho_tx ** i
    Rtx = toeplitz(Rtx_vec)
    if Mr == Nt and rho_tx == rho_rx:
        Rrx = Rtx
    else:
        Rrx_vec = np.ones(Mr)
        for i in range(1, Mr):
            Rrx_vec[i] = rho_rx ** i
        Rrx = toeplitz(Rrx_vec)

    # another way of constructing kronecker model
    # C = cholesky(np.kron(Rtx,Rrx))    # complex correlation
    # C = sqrtm(np.sqrt(np.kron(Rtx, Rrx)))  # power field correlation--what's an equivalent model?
    # return C

    sqrtRtx = sqrtm(Rtx)  # sqrt decomposition for power field

    if Mr == Nt and rho_tx == rho_rx:
        sqrtRrx = sqrtRtx
    else:
        sqrtRrx = sqrtm(Rrx)

    return sqrtRtx, sqrtRrx


def detector(sys, H, x, y, sigma2, MSE, model=None):
    detect_type = sys.detect_type
    randomized = sys.randomized
    if sys.use_OFDM:
        Mr, Nt = sys.Mr * sys.K, sys.Nt * sys.K
    else:
        Mr, Nt = sys.Mr, sys.Nt
    mu = sys.mu
    T = sys.T
    sess, prob, x_hat_net = sys.sess, sys.prob, sys.x_hat_net
    if detect_type == 'CG_OAMP_NET' or detect_type == 'OAMP_NET':  # use NET
        y = y.reshape((1, 2 * Mr, 1)).astype(np.float32)
        H_bar = H.reshape((1, 2 * Mr, 2 * Nt)).astype(np.float32)
        sigma2 = sigma2 * np.ones((1, 1, 1), dtype=np.float32)
        if detect_type == 'CG_OAMP_NET':  # CG_OAMP_NET
            HHT = H @ H.T
            # use complex value with half dimension for faster eigendecomposition
            C = (HHT[:Mr, :Mr] + 1j * HHT[Mr:2 * Mr, :Mr])  # .reshape((1,Mr,Mr)).astype(np.complex64)
            eigval = np.linalg.eigvalsh(C).astype(np.float32).reshape((1, Mr, 1))
            x_hat = sess.run(x_hat_net, feed_dict={prob.y_: y,
                                                   prob.x_: np.zeros((1, 2 * Nt, 1), dtype=np.float32), prob.H_: H_bar,
                                                   prob.sigma2_: sigma2, prob.sample_size_: 1,
                                                   prob.eigvalue_: eigval})
            # prob.sigma2_: sigma2, prob.sample_size_: 1, prob.C_: C})
        else:  # OAMP-NET
            x_hat = sess.run(x_hat_net, feed_dict={prob.y_: y,
                                                   prob.x_: np.zeros((1, 2 * Nt, 1), dtype=np.float32), prob.H_: H_bar,
                                                   prob.sigma2_: sigma2, prob.sample_size_: 1})
    elif detect_type == 'CG_OAMP':
        x_hat, _, mse = CG_OAMP(x, H, y, sigma2 / 2, I=50, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'OAMP':
        x_hat, mse = OAMP(x, H, y, sigma2 / 2, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'MAMP':
        s = la.svd(H, compute_uv=False)
        lambda_dag = (max(s) ** 2 + min(s) ** 2) / 2
        x_hat, mse = MAMP(x, H, s, y, lambda_dag, sigma2 / 2, L=3, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'ZF':  # ZF
        HT = H.T
        x_hat = la.inv(HT @ H) @ HT @ y
        MSE += np.mean(abs(x - x_hat) ** 2)
    elif detect_type == 'MMSE':  # MMSE
        HT = H.T
        x_hat = la.inv(HT @ H + sigma2 / 2 * np.eye(2 * Nt)) @ HT @ y
        MSE += np.mean(abs(x - x_hat) ** 2)
    elif detect_type == 'EP_real_v1':
        x_hat, mse = EP_real_v1(x, H, y, sigma2 / 2, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'EP_real_v2':
        x_hat, mse = EP_real_v2(x, H, y, sigma2 / 2, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'EP_real_v3':
        x_hat, mse = EP_real_v3(x, H, y, sigma2 / 2, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'GEPNet':
        x = x.reshape((1, 2 * Nt, 1))
        y = y.reshape((1, 2 * Mr, 1))
        H = H.reshape((1, 2 * Mr, 2 * Nt))
        _, x_hat, _ = model(x, y, H, sigma2)
        mse = np.mean(abs(x_hat - x) ** 2)
        MSE += np.array([mse])
        x_hat = x_hat.numpy().reshape(-1, 1)
    elif detect_type == 'ML':
        x_hat = mlSolver(y, H, mu).reshape(-1, 1)
        MSE += np.mean((x - x_hat) ** 2)
    elif detect_type == 'gibbs_sampling':
        x_hat = gibbs_sampling(x, H, y, sigma2 / 2, mu=mu, randomized=randomized, alpha=sys.alpha)
    elif detect_type == 'AMP':
        x_hat, mse = amp(x, H, y, sigma2 / 2, t_max=T, mu=mu)
        MSE += mse
    elif detect_type == 'MHGD':
        x_hat, mse = mhgd(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                          mmse_init=sys.mmse_init)
        MSE += mse
    elif detect_type == 'MHGD_PARA':
        x_hat, mse = mhgd_para(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                               mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size)
        MSE += mse
    elif detect_type == 'langevin':
        x_hat, mse = langevin(sys, x, H, y, sigma2, sys.lang)
    else:
        raise RuntimeError('The selected detector does not exist!')

    return x_hat, MSE


def mlSolver(y, h_real, mu):
    # status = []
    m, n = h_real.shape[0], h_real.shape[1]
    model = Model('mimo')
    M = 2 ** (mu // 2)
    sigConst = np.linspace(-M + 1, M - 1, M)
    sigConst /= np.sqrt((sigConst ** 2).mean())
    sigConst /= np.sqrt(2.)  # Each complex transmitted signal will have two parts
    z = model.addVars(n, M, vtype=GRB.BINARY, name='z')
    s = model.addVars(n, ub=max(sigConst) + .1, lb=min(sigConst) - .1, name='s')
    e = model.addVars(m, ub=200.0, lb=-200.0, vtype=GRB.CONTINUOUS, name='e')
    model.update()

    ### Constraints and variables definitions
    # define s[i]
    for i in range(n):
        model.addConstr(s[i] == quicksum(z[i, j] * sigConst[j] for j in range(M)))
    # constraint on z[i,j]
    model.addConstrs((z.sum(j, '*') == 1 for j in range(n)), name='const1')
    # define e
    for i in range(m):
        e[i] = quicksum(h_real[i, j] * s[j] for j in range(n)) - y[i]

    ### define the objective function
    obj = e.prod(e)
    model.setObjective(obj, GRB.MINIMIZE)
    model.Params.logToConsole = 0
    model.setParam('TimeLimit', 100)
    model.update()

    model.optimize()

    # retrieve optimization result
    solution = model.getAttr('X', s)
    # status.append(model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL)
    # print(GRB.OPTIMAL, model.getAttr(GRB.Attr.Status))
    if model.getAttr(GRB.Attr.Status) == 9:
        print(np.linalg.cond(h_real))
    x_hat = []
    for num in solution:
        x_hat.append(solution[num])
    return np.array(x_hat)


def state_evolution(sysin, SNR=40):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    T = sysin.T
    H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                               + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
    if channel_type == 'corr':  # Correlated MIMO channel
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
        H = sqrtRrx @ H @ sqrtRtx
    H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                        np.concatenate((np.imag(H), np.real(H)), axis=1)))
    s = la.svd(H, compute_uv=False)
    MSE = EP_U_SE(H, s, Nt / Mr * 10 ** (-(SNR) / 10), T=T, mu=mu)

    return MSE


def nr_corr_channel(Mr, Nt, corr_level, a):
    if corr_level == 'low':
        alpha, beta = 0, 0
    elif corr_level == 'med':
        alpha, beta = 0.3, 0.9
    elif corr_level == 'meda':
        alpha, beta = 0.3, 0.3874
    else:
        alpha, beta = 0.9, 0.9

    # Generate correlation matrix of NodeB side
    if Mr == 1:
        renb = 1
    elif Mr == 2:
        renb = toeplitz([1, alpha])
    elif Mr == 4:
        renb = toeplitz([1, alpha ** (1 / 9), alpha ** (4 / 9), alpha])
    elif Mr == 8:
        renb = toeplitz([1, alpha ** (1 / 49), alpha ** (4 / 49), alpha ** (9 / 49),
                         alpha ** (16 / 49), alpha ** (25 / 49), alpha ** (36 / 49), alpha])
    else:
        renb = np.eye(Mr)

    # Generate correlation matrix of UE side
    if Nt == 1:
        rue = 1
    elif Nt == 2:
        rue = toeplitz([1, beta])
    elif Nt == 4:
        rue = toeplitz([1, beta ** (1 / 9), beta ** (4 / 9), beta])
    elif Nt == 8:
        rue = toeplitz([1, beta ** (1 / 49), beta ** (4 / 49), beta ** (9 / 49),
                        beta ** (16 / 49), beta ** (25 / 49), beta ** (36 / 49), beta])
    else:
        rue = np.eye(Nt)

    # combined spatial correlation matrix
    rspat = np.kron(renb, rue)

    # "a" is a scaling factor such that the smallest value is used to make Rspat a positive semi-definite
    rspat = (rspat + a * np.eye(Mr * Nt)) / (1 + a)

    return rspat


def channel_est(sysin, snr):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    Np = Nt  # the number of pilot vectors
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    print('calculate covariance and LMMSE weight matrix for CE')
    num = 10000
    rhh = np.zeros((Mr * Nt, Mr * Nt), dtype=complex)
    for n in range(num):
        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx
        h = H.reshape(-1, 1)
        rhh += h @ np.conj(h.T)
    rhh /= num
    xp = dft(Np)[:Nt, :]  # orthogonal pilots (nt, np)
    sigma2 = 10 ** (-snr / 10)
    yp = np.zeros((Mr, Np), dtype=complex)
    wlmmse = lmmse_ce(xp, yp, sigma2, rhh)
    print('end')
    return wlmmse, xp
