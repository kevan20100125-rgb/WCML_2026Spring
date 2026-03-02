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
from .mcmc import gibbs_sampling, gibbs_sampling_para
from .swamp import amp, amp_mimo
from .MHGD import mhgd, mhgd_para, mhngd_para, mala_para, cg, dmhgd, mhgd_para_temp, dmhngd, mhgd_real
from .MHCG import mhcg, mhcg_para
from .HMC import hmc
from .langevin_numpy import LangevinNumpy, langevin, LangevinNumpyPara, langevin_para
from commpy import QAMModem
from .distributed_det import dcg, dep_real, dnewton
from .SGD_DAISY import rls, sgd, asgd
from .kbest import kbest
from .SGLD import sgld
from .discrete_mcmc import dis_ula, dis_mala, dis_pavg

from gurobipy import *
import scipy.io as sio

pi = math.pi


def MIMO_detection_simulate(model, sysin, SNR=40):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    csi = sysin.csi
    sysin.acc, sysin.avg_ng, sysin.avg_ns, sysin.avg_nz = 0, 0, 0, 0
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    ser = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    Hdataset = None
    if mu > 6:
        modem = QAMModem(2 ** mu)
    else:
        modem = None
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    elif channel_type == '3gpp':
        Hdataset = sio.loadmat('data/H_3gpp_20mhz.mat')['H1']
    elif channel_type == '3gppo2i':  # from [Zilberstein, 2022]
        Hdataset = sio.loadmat('data/H_bank')['H_bank']
    elif channel_type == 'OTA':
        Hdataset = load_data(num_of_data=10)  # 10*14*1620 channel matrices
    elif channel_type == 'WLAN':
        Hdataset = sio.loadmat('data/TGN_D_' + str(Mr) + 'X' + str(Nt) + '.mat')['H']

    if csi == 2:
        Np = sysin.Np
        wlmmse, xp = channel_est(sysin, SNR, orth=sysin.orth)
    if sysin.colored:
        # sysin.cov = toeplitz(0.3 ** np.arange(Mr))
        sysin.cov = np.load('data/covar.npy')
        sysin.cov_raw = sysin.cov.copy()
        L = cholesky(sysin.cov, lower=True)
    else:
        sysin.cov, L = None, None
    complex_dict = {'MHGD', 'MHGD_PARA', 'MHGD_PARA_TEMP', 'MHNGD_PARA', 'MHCG', 'MHCG_PARA', 'MALA_PARA',
                    'DMHGD', 'DMHGDNet', 'DCG', 'DNT', 'RLS', 'SGD', 'ASGD', 'DMHNGD', 'SGLD'}

    if sysin.detect_type == 'langevin':
        # create noise vector for Langevin
        sigma_gaussian = np.exp(np.linspace(np.log(sysin.sigma_0), np.log(sysin.sigma_L),
                                            sysin.n_sigma_gaussian))

        # create model
        sysin.lang = LangevinNumpy(sigma_gaussian, sysin.n_sample_init, sysin.step_size, sysin.mu)
    elif sysin.detect_type == 'langevin_para':
        # create noise vector for Langevin
        sigma_gaussian = np.exp(np.linspace(np.log(sysin.sigma_0), np.log(sysin.sigma_L),
                                            sysin.n_sigma_gaussian))

        # create model
        sysin.lang = LangevinNumpyPara(sigma_gaussian, sysin.n_sample_init, sysin.step_size, sysin.mu, sysin.n_traj)

    norm = np.sqrt(1)
    # num_trail = 10000
    # for i in range(num_trail):
    while True:
        # generate bits and modulate
        # if channel_type == 'OTA' or channel_type == 'WLAN':
        np.random.seed(count)  # choose the same channel
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu, modem=modem)
        x = bits_mod.reshape(Nt, 1) / norm

        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx
        elif channel_type == '3gpp':
            idx1 = np.random.randint(low=0, high=Hdataset.shape[0])
            idx2, idx3 = np.random.randint(low=0, high=Hdataset.shape[3]), np.random.randint(low=0, high=Hdataset.shape[4])
            H = Hdataset[idx1, :, :, idx2, idx3].T  # (64, 8)
            H = H / np.sqrt(np.sum(abs(H) ** 2, axis=0))  # column normalization (condition number drops a little)
            # from [ANDREA SCOTTI, PhD thesis], [GOWDA, MHGD's remove channel gain],
            # make y=x1h1+...+xNhN the same Rx energy as the Tx (over random H & x)
        elif channel_type == '3gppo2i':  # randomly select nt users from 100 users
            if Nt > Hdataset.shape[2]:
                raise RuntimeError('Greater than the maximum number of users!')
            idx1 = np.random.randint(low=0, high=Hdataset.shape[0])  # select the subcarrier
            idx2 = np.random.permutation(np.arange(Hdataset.shape[2]))[:Nt]  # select the nt users
            H = Hdataset[idx1, :, idx2].T  # (64, nt)
            H = H / np.sqrt(np.sum(abs(H) ** 2, axis=0))  # column normalization (condition number drops a little)
        elif channel_type == 'OTA':
            if Mr > 20 or Nt > 16:
                raise RuntimeError('Out of size!')
            idx1 = np.random.randint(low=0, high=Hdataset.shape[0])  # symbol index in a frame
            idx2 = np.random.randint(low=0, high=Hdataset.shape[1])  # subcarrier index
            H = Hdataset[idx1, idx2][:Mr, :Nt]
            H = H / np.sqrt(np.sum(abs(H) ** 2, axis=0))  # column normalization (condition number drops a little)
        elif channel_type == 'WLAN':
            idx1 = np.random.randint(low=0, high=Hdataset.shape[0])  # sample index
            idx2 = np.random.randint(low=0, high=Hdataset.shape[1])  # path index
            H = Hdataset[idx1, idx2, :, :]  # original ntxnr in matlab, not need to transpose
            H = H / np.sqrt(np.sum(abs(H) ** 2, axis=0))  # column normalization (condition number drops a little)

        # channel input & output
        y = H @ x

        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-(SNR) / 10)  # noise power per receive ant.; average SNR per receive ant.
        if sysin.colored:
            noise = L @ (np.random.randn(Mr, 1) + 1j * np.random.randn(Mr, 1)) / np.sqrt(2)
            # normalize so that noise norm is sqrt(Mr * sigma2)
            # sysin.cov = sysin.cov_raw * (np.sqrt(Mr * sigma2) / la.norm(noise)) ** 2
            # noise = np.sqrt(Mr * sigma2) * noise / la.norm(noise)
        else:  # add AWGN noise
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                           + 1j * np.random.randn(Mr, 1))
        y = y + noise

        if csi == 2:
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, Np) + 1j * np.random.randn(Mr, Np))
            yp = H @ xp + noise
            yp_vec = yp.reshape(-1, 1)
            h_hat = wlmmse @ yp_vec
            H = h_hat.reshape(Mr, Nt)
        elif csi == 1:
            H = H + np.sqrt(sysin.sigma2h / 2) * (np.random.randn(Mr, Nt) + 1j * np.random.randn(Mr, Nt))

        if sysin.detect_type not in complex_dict:
            # convert complex into real
            x = np.concatenate((np.real(x), np.imag(x)))
            H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                                np.concatenate((np.imag(H), np.real(H)), axis=1)))
            y = np.concatenate((np.real(y), np.imag(y)))

        sta = time.time()
        # if count == 73:
        #     print('')
        x_hat, MSE = detector(sysin, H, x, y, sigma2, MSE, model, modem=modem)
        end = time.time()

        if sysin.detect_type not in complex_dict:
            # back into np.complex64
            x_hat = x_hat.reshape((2, Nt))
            x_hat = x_hat[0, :] + 1j * x_hat[1, :]

        # Demodulate
        x_hat_demod = QAM_Demodulation(x_hat * norm, mu, modem)

        total_time += (end - sta)

        # calculate BER
        err_bits = np.sum(np.not_equal(x_hat_demod, bits))
        total_err_bits += err_bits
        total_bits += mu * Nt
        count = count + 1
        if err_bits > 0:
            ser += calc_ser(x_hat_demod, bits, Nt, mu)
            sys.stdout.write(
                '\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f} SER={SER:.6f} ng={ng:.3f} ns={ns:.3f} '
                'nz={nz:.3f} acc={acc:.3f}'
                .format(teb=total_err_bits, tb=total_bits, BER=total_err_bits / total_bits,
                        SER=ser / count / Nt, ng=sysin.avg_ng / count, ns=sysin.avg_ns / count, nz=sysin.avg_nz / count,
                        acc=sysin.acc / count))
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
            print("Average ng:", sysin.avg_ng / count)
            print("Average ns:", sysin.avg_ns / count)
            print("Average nz:", sysin.avg_nz / count)
            print("Average acc rate:", sysin.acc / count)
            break

    # print("time:",total_time/1000)
    return ber, ser, np.array([total_err_bits, total_bits, sysin.avg_ns / count, sysin.avg_nz / count])


def MIMO_detection_batch(sysin, SNR=40, batch_size=64):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    sess, prob, x_hat_net, mse = sysin.sess, sysin.prob, sysin.x_hat_net, sysin.mse
    csi = sysin.csi
    x_, y_, H_, sigma2_, bs_, label_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.sample_size_, prob.label_
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    ser = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    Hdataset = None
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    elif channel_type == '3gpp':
        Hdataset = sio.loadmat('data/H_3gpp_20mhz.mat')['H1']
    if csi == 2:
        Np = Nt
        wlmmse, xp = channel_est(sysin, SNR)
    norm = np.sqrt(1)
    # if sysin.detect_type not in {'MHGDNet', 'MHCGNet'}:
    #     H_batch = np.zeros((batch_size, 2 * Mr, 2 * Nt))
    #     x_batch = np.zeros((batch_size, 2 * Nt, 1))
    #     y_batch = np.zeros((batch_size, 2 * Mr, 1))
    #     bits_batch = np.zeros((batch_size, Nt * mu), dtype=int)
    # else:
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
        elif channel_type == '3gpp':
            idx1 = np.random.randint(low=0, high=Hdataset.shape[0])
            idx2, idx3 = np.random.randint(low=0, high=Hdataset.shape[3]), np.random.randint(low=0, high=Hdataset.shape[4])
            H = Hdataset[idx1, :, :, idx2, idx3].T  # (64, 8)
            H = H / np.sqrt(np.sum(abs(H) ** 2, axis=0))  # column normalization

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

        # if sysin.detect_type not in {'MHGDNet', 'MHCGNet'}:
            # convert complex into real
            # x = np.concatenate((np.real(x), np.imag(x)))
            # H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
            #                     np.concatenate((np.imag(H), np.real(H)), axis=1)))
            # y = np.concatenate((np.real(y), np.imag(y)))

        # stack
        H_batch[count % batch_size] = H
        x_batch[count % batch_size] = x
        y_batch[count % batch_size] = y
        bits_batch[count % batch_size] = bits

        count = count + 1
        if count % batch_size == 0:
            sta = time.time()
            if sysin.detect_type == 'AMPNet':
                x_hat_batch, mse_batch = sess.run((x_hat_net, mse), feed_dict={y_: y_batch,
                                                                               x_: x_batch, H_: H_batch,
                                                                               bs_: batch_size,
                                                                               sigma2_: sigma2 * np.ones(
                                                                                   (batch_size, 1, 1),
                                                                                   dtype=np.float64),
                                                                               label_: np.zeros((batch_size, 2 ** (mu // 2), 2 * Nt))},
                                                                               )
            else:
                x_hat_batch, mse_batch = sess.run((x_hat_net, mse), feed_dict={y_: y_batch,
                                                                               x_: x_batch, H_: H_batch,
                                                                               bs_: batch_size,
                                                                               sigma2_: sigma2 * np.ones(
                                                                                   (batch_size, 1, 1),
                                                                                   dtype=np.float64)})
            # x_hat: (bs, 2*Nt, 1)
            end = time.time()
            # mse = np.mean(abs(x_hat_batch - x_batch) ** 2)
            # MSE += np.array([mse])
            MSE += mse_batch

            for m in range(batch_size):
                x_hat = x_hat_batch[m]
                # if sysin.detect_type not in {'MHGDNet', 'MHCGNet'}:
                #     # back into np.complex64
                #     x_hat = x_hat.reshape((2, Nt))
                #     x_hat = x_hat[0, :] + 1j * x_hat[1, :]

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


def sample_gen(trainSet, ts, vs, training_flag=True, fixed_channel=False, H_fixed=None, label=False):
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
    indicator_ = np.zeros((ts, 2 ** (mu // 2), 2 * Nt), dtype=int)
    # generate development samples:
    Hval_ = np.zeros((vs * Mr, Nt), dtype=complex)
    xval_ = np.zeros((vs * Nt, 1), dtype=complex)
    yval_ = np.zeros((vs * Mr, 1), dtype=complex)
    sigma2val_ = np.zeros((vs, 1))
    indicatorval_ = np.zeros((vs, 2 ** (mu // 2), 2 * Nt), dtype=int)

    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if channel_type == 'nr':
        rspat = nr_corr_channel(Mr, Nt, 'meda', a=0.0)
        rherm = cholesky(rspat).T

    for i in range(ts + vs):
        if trainSet.snr == 'varying_':
            SNR = np.random.randint(low=15, high=26)
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1)
        # Rayleigh MIMO channel
        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt) +
                                   1j * np.random.randn(Mr, Nt))
        if fixed_channel:
            H = H_fixed.copy()
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
            if label:
                indicator_[i] = indicator(bits, mu)
        else:
            Hval_[Mr * (i - ts):Mr * (i - ts + 1)] = H
            xval_[Nt * (i - ts):Nt * (i - ts + 1)] = x
            yval_[Mr * (i - ts):Mr * (i - ts + 1)] = y
            sigma2val_[i - ts] = sigma2
            if label:
                indicatorval_[i - ts] = indicator(bits, mu)
    # reshape
    H_ = H_.reshape(ts, Mr, Nt)
    x_ = x_.reshape(ts, Nt, 1)
    y_ = y_.reshape(ts, Mr, 1)
    sigma2_ = sigma2_.reshape(ts, 1, 1)
    Hval_ = Hval_.reshape(vs, Mr, Nt)
    xval_ = xval_.reshape(vs, Nt, 1)
    yval_ = yval_.reshape(vs, Mr, 1)
    sigma2val_ = sigma2val_.reshape(vs, 1, 1)

    if label:
        return y_, x_, H_, sigma2_, indicator_, yval_, xval_, Hval_, sigma2val_, indicatorval_
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


def detector(sys, H, x, y, sigma2, MSE, model=None, modem=None):
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
    elif detect_type == 'CG':
        HT = H.T
        x_hat = cg(np.zeros_like(x), xi=HT @ H + sigma2 * np.eye(2 * Nt), y=HT @ y, iter=T)
        MSE += np.mean(abs(x - x_hat) ** 2)
    elif detect_type == 'DCG':
        x_hat = dcg(x, H, y, sys.c, sigma2, iter=T, det='MMSE')
        MSE += np.mean(abs(x - x_hat) ** 2)
    elif detect_type == 'DNT':
        x_hat, mse = dnewton(x, H, y, sys.c, iter=T, hessian_approx=sys.hessian_approx)
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
    elif detect_type == 'DEP_real':
        x_hat, mse = dep_real(x, H, y, sys.c, sigma2 / 2, T=T, mu=mu)
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
    elif detect_type == 'kbest':
        x_hat, mse = kbest(x, H, y, mu=mu, k=sys.k)
        MSE += mse
    elif detect_type == 'gibbs_sampling':
        x_hat, mse = gibbs_sampling(x, H, y, sigma2 / 2, mu=mu, randomized=randomized, alpha=sys.alpha, iter=sys.samples)
        MSE += mse
    elif detect_type == 'gibbs_sampling_para':
        x_hat, mse = gibbs_sampling_para(x, H, y, sigma2 / 2, mu=mu, randomized=randomized, alpha=sys.alpha,
                                         iter=sys.samples, samplers=sys.samplers)
        MSE += mse
    elif detect_type == 'AMP':
        x_hat, mse = amp(x, H, y, sigma2 / 2, t_max=T, mu=mu)
        MSE += mse
    elif detect_type == 'SGLD':
        x_hat, mse = sgld(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers)
        MSE += mse
    elif detect_type == 'MHGD':
        x_hat, mse = mhgd(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                          mmse_init=sys.mmse_init)
        MSE += mse
    elif detect_type == 'MHNGD_PARA':
        x_hat, mse, ng, ns, nz = mhngd_para(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, ng=sys.ng,
                                            mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                                            adaptive_lr=sys.adaptive_lr,
                                            quantize=sys.quantize, post=sys.post, sur=sys.sur, early_stop=sys.es,
                                            colored=sys.colored, noise_cov=sys.cov,
                                            constellation_norm=modem.constellation if mu > 6 else None,
                                            restart=sys.restart, hessian_approx=sys.hessian_approx)
        MSE += mse
        sys.avg_ng += ng
        sys.avg_ns += ns
        sys.avg_nz += nz
    elif detect_type == 'MHGD_PARA':
        x_hat, mse, acc = mhgd_para(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                               mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                               constellation_norm=modem.constellation if mu > 6 else None, hessian_approx=sys.hessian_approx)
        MSE += mse
        sys.acc += acc
    elif detect_type == 'MHGD_REAL':
        x_hat, mse, acc = mhgd_real(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                               mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                               constellation_norm=modem.constellation if mu > 6 else None, hessian_approx=sys.hessian_approx)
        MSE += mse
        sys.acc += acc
    elif detect_type == 'MHGD_PARA_TEMP':
        x_hat, mse = mhgd_para_temp(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers,
                                    lr_approx=sys.lr_approx, mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size)
        MSE += mse
    elif detect_type == 'DULA':
        x_hat, mse = dis_ula(x, H, y, sigma2 / 2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                             mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                             constellation_norm=modem.constellation if mu > 6 else None, hessian_approx=sys.hessian_approx)
        MSE += mse
    elif detect_type == 'DMALA':
        x_hat, mse, acc = dis_mala(x, H, y, sigma2 / 2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                              mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                              constellation_norm=modem.constellation if mu > 6 else None, hessian_approx=sys.hessian_approx)
        MSE += mse
        sys.acc += acc
    elif detect_type == 'DPAVG':
        x_hat, mse, acc = dis_pavg(x, H, y, sigma2 / 2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                              mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                              constellation_norm=modem.constellation if mu > 6 else None, hessian_approx=sys.hessian_approx)
        MSE += mse
        sys.acc += acc
    elif detect_type == 'DMHGD':
        x_hat, mse, acc, ns = dmhgd(x, H, y, sys.c, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers,
                                    lr_approx=sys.lr_approx, mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                                    hessian_approx=sys.hessian_approx, early_stop=sys.es)
        MSE += mse
        sys.acc += acc
        sys.avg_ns += ns
    elif detect_type == 'DMHNGD':
        x_hat, mse, ng, ns, nz = dmhngd(x, H, y, sys.c, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, ng=sys.ng,
                                        mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size,
                                        quantize=sys.quantize, post=sys.post, early_stop=sys.es,
                                        hessian_approx=sys.hessian_approx,
                                        central_gd=(True if sys.channel_type == '3gpp' else False))
        MSE += mse
        sys.avg_ng += ng
        sys.avg_ns += ns
        sys.avg_nz += nz
    elif detect_type == 'DMHGDNet':
        x_hat, mse = dmhgd(x, H, y, sys.c, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers,
                           lr_approx=sys.lr_approx, mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size, net=True)
        MSE += mse
    elif detect_type == 'MHCG':
        x_hat, mse = mhcg(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, lr_approx=sys.lr_approx,
                          mmse_init=sys.mmse_init)
        MSE += mse
    elif detect_type == 'MHCG_PARA':
        x_hat, mse = mhcg_para(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers,
                               mmse_init=sys.mmse_init, vec_step_size=sys.vec_step_size)
        MSE += mse
    elif detect_type == 'MALA_PARA':
        x_hat, mse = mala_para(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, mmse_init=sys.mmse_init)
        MSE += mse
    elif detect_type == 'HMC':
        x_hat, mse = hmc(x, H, y, sigma2 / 2, mu=mu, iter=sys.samples, samplers=sys.samplers, leapfrogs=sys.leapfrogs,
                         mmse_init=sys.mmse_init)
    elif detect_type == 'langevin':
        x_hat, mse = langevin(sys, x, H, y, sigma2, sys.lang)
        MSE += mse
    elif detect_type == 'langevin_para':
        x_hat, mse = langevin_para(sys, x, H, y, sigma2, sys.lang)
        MSE += mse
    elif detect_type == 'RLS':
        x_hat, mse = rls(x, H, y)
        MSE += mse
    elif detect_type == 'SGD':
        x_hat, mse = sgd(x, H, y)
        MSE += mse
    elif detect_type == 'ASGD':
        x_hat, mse = asgd(x, H, y)
        MSE += mse
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


def channel_est(sysin, snr, orth=True):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    Np = sysin.Np  # the number of pilot vectors
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
    if orth:
        if Np >= Nt:
            xp = dft(Np)[:Nt, :]  # orthogonal pilots (nt, np)
        else:
            xp = dft(Nt)[:, :Np]
    else:
        xp = 1 / np.sqrt(2) * (2 * np.random.binomial(1, 0.5, size=(Nt, Np)) - 1 +
                               1j * (2 * np.random.binomial(1, 0.5, size=(Nt, Np)) - 1))  # QPSK pilots
    sigma2 = 10 ** (-snr / 10)
    yp = np.zeros((Mr, Np), dtype=complex)
    wlmmse = lmmse_ce(xp, yp, sigma2, rhh)
    print('end')
    return wlmmse, xp


# load OTA data
def load_data(num_of_data=10):
    test_path = "data/OTA_S4_WALKING"
    H_test = sio.loadmat(test_path + "/HPDSCH_iqdata1.mat")['HPDSCH']  # (ns,1620) each element is an (20,16) array
    for i in range(1, num_of_data):  # stack
        H_test = np.vstack((H_test, sio.loadmat(
            test_path + "/HPDSCH_iqdata" +
            str(i + 1) + ".mat")['HPDSCH']))

    return H_test

