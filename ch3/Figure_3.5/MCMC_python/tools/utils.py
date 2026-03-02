#!/usr/bin/python
from __future__ import division
import numpy as np
import math
import os
import time
import numpy.linalg as la
from commpy import QAMModem

sqrt = np.sqrt
pi = math.pi

_QPSK_mapping_table = {
    (0,1): (-1+1j,), (1,1): (1+1j,),
    (0,0): (-1-1j,), (1,0): (1-1j,)
}

_QPSK_demapping_table = {v: k for k, v in _QPSK_mapping_table.items()}

_QPSK_Constellation = np.array([[-1-1j], [-1+1j],
                                [1-1j], [1+1j]])

_16QAM_mapping_table = {
    (0,0,1,0): (-3+3j,), (0,1,1,0): (-1+3j,), (1,1,1,0): (1+3j,), (1,0,1,0): (3+3j,),
    (0,0,1,1): (-3+1j,), (0,1,1,1): (-1+1j,), (1,1,1,1): (1+1j,), (1,0,1,1): (3+1j,),
    (0,0,0,1): (-3-1j,), (0,1,0,1): (-1-1j,), (1,1,0,1): (1-1j,), (1,0,0,1): (3-1j,),
    (0,0,0,0): (-3-3j,), (0,1,0,0): (-1-3j,), (1,1,0,0): (1-3j,), (1,0,0,0): (3-3j,)
}

_16QAM_demapping_table = {v: k for k, v in _16QAM_mapping_table.items()}

_16QAM_Constellation = np.array([-3-3j,-3-1j,-3+3j,-3+1j,
                                -1-3j,-1-1j,-1+3j,-1+1j,
                                +3-3j,+3-1j,+3+3j,+3+1j,
                                +1-3j,+1-1j,+1+3j,+1+1j]).reshape(-1,1)

_64QAM_mapping_table = {
    (0,0,0,1,0,0): (-7+7j,), (0,0,1,1,0,0): (-5+7j,), (0,1,1,1,0,0): (-3+7j,), (0,1,0,1,0,0): (-1+7j,), (1,1,0,1,0,0): (1+7j,), (1,1,1,1,0,0): (3+7j,), (1,0,1,1,0,0): (5+7j,), (1,0,0,1,0,0): (7+7j,),
    (0,0,0,1,0,1): (-7+5j,), (0,0,1,1,0,1): (-5+5j,), (0,1,1,1,0,1): (-3+5j,), (0,1,0,1,0,1): (-1+5j,), (1,1,0,1,0,1): (1+5j,), (1,1,1,1,0,1): (3+5j,), (1,0,1,1,0,1): (5+5j,), (1,0,0,1,0,1): (7+5j,),
    (0,0,0,1,1,1): (-7+3j,), (0,0,1,1,1,1): (-5+3j,), (0,1,1,1,1,1): (-3+3j,), (0,1,0,1,1,1): (-1+3j,), (1,1,0,1,1,1): (1+3j,), (1,1,1,1,1,1): (3+3j,), (1,0,1,1,1,1): (5+3j,), (1,0,0,1,1,1): (7+3j,),
    (0,0,0,1,1,0): (-7+1j,), (0,0,1,1,1,0): (-5+1j,), (0,1,1,1,1,0): (-3+1j,), (0,1,0,1,1,0): (-1+1j,), (1,1,0,1,1,0): (1+1j,), (1,1,1,1,1,0): (3+1j,), (1,0,1,1,1,0): (5+1j,), (1,0,0,1,1,0): (7+1j,),
    (0,0,0,0,1,0): (-7-1j,), (0,0,1,0,1,0): (-5-1j,), (0,1,1,0,1,0): (-3-1j,), (0,1,0,0,1,0): (-1-1j,), (1,1,0,0,1,0): (1-1j,), (1,1,1,0,1,0): (3-1j,), (1,0,1,0,1,0): (5-1j,), (1,0,0,0,1,0): (7-1j,),
    (0,0,0,0,1,1): (-7-3j,), (0,0,1,0,1,1): (-5-3j,), (0,1,1,0,1,1): (-3-3j,), (0,1,0,0,1,1): (-1-3j,), (1,1,0,0,1,1): (1-3j,), (1,1,1,0,1,1): (3-3j,), (1,0,1,0,1,1): (5-3j,), (1,0,0,0,1,1): (7-3j,),
    (0,0,0,0,0,1): (-7-5j,), (0,0,1,0,0,1): (-5-5j,), (0,1,1,0,0,1): (-3-5j,), (0,1,0,0,0,1): (-1-5j,), (1,1,0,0,0,1): (1-5j,), (1,1,1,0,0,1): (3-5j,), (1,0,1,0,0,1): (5-5j,), (1,0,0,0,0,1): (7-5j,),
    (0,0,0,0,0,0): (-7-7j,), (0,0,1,0,0,0): (-5-7j,), (0,1,1,0,0,0): (-3-7j,), (0,1,0,0,0,0): (-1-7j,), (1,1,0,0,0,0): (1-7j,), (1,1,1,0,0,0): (3-7j,), (1,0,1,0,0,0): (5-7j,), (1,0,0,0,0,0): (7-7j,)
}

_64QAM_demapping_table = {v: k for k, v in _64QAM_mapping_table.items()}

_64QAM_Constellation = np.array([ -7-7j,-7-5j,-7-1j,-7-3j,-7+7j,-7+5j,-7+1j,-7+3j,
                              -5-7j,-5-5j,-5-1j,-5-3j,-5+7j,-5+5j,-5+1j,-5+3j,
                              -1-7j,-1-5j,-1-1j,-1-3j,-1+7j,-1+5j,-1+1j,-1+3j,
                              -3-7j,-3-5j,-3-1j,-3-3j,-3+7j,-3+5j,-3+1j,-3+3j,
                              +7-7j,+7-5j,+7-1j,+7-3j,+7+7j,+7+5j,+7+1j,+7+3j,
                              +5-7j,+5-5j,+5-1j,+5-3j,+5+7j,+5+5j,+5+1j,+5+3j,
                              +1-7j,+1-5j,+1-1j,+1-3j,+1+7j,+1+5j,+1+1j,+1+3j,
                              +3-7j,+3-5j,+3-1j,+3-3j,+3+7j,+3+5j,+3+1j,+3+3j]).reshape(-1,1)

_QPSK_onehot_mapping_table = {  # -1 +1
    (0): (0, 1), (1): (1, 0)
}

_16QAM_onehot_mapping_table = {   # -3 -1 +1 +3
    (0, 0): (0, 0, 0, 1), (0, 1): (0, 0, 1, 0),
    (1, 1): (0, 1, 0, 0), (1, 0): (1, 0, 0, 0)
}

_64QAM_onehot_mapping_table = {  # -7 -5 -3 -1 +1 +3 +5 +7
    (0, 0, 0): (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 1): (0, 0, 0, 0, 0, 0, 1, 0),
    (0, 1, 1): (0, 0, 0, 0, 0, 1, 0, 0), (0, 1, 0): (0, 0, 0, 0, 1, 0, 0, 0),
    (1, 1, 0): (0, 0, 0, 1, 0, 0, 0, 0), (1, 1, 1): (0, 0, 1, 0, 0, 0, 0, 0),
    (1, 0, 1): (0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0): (1, 0, 0, 0, 0, 0, 0, 0)
}

sq2 = sqrt(2)
sq10 = sqrt(10)
sq42 = sqrt(42)


def indicator(bits, mu):
    indicator = []
    for i in range(int(len(bits) / (mu//2) )):
        if mu == 2:
            indicator.append(list(_QPSK_onehot_mapping_table.get(bits[i])))  # shape(-1, 2)
        elif mu == 4:
            indicator.append(list(_16QAM_onehot_mapping_table.get(tuple(bits[2*i:2*(i+1)]))))  # shape(-1, 4)
        else:
            indicator.append(list(_64QAM_onehot_mapping_table.get(tuple(bits[3 * i:3 * (i + 1)]))))  # shape(-1, 8)
    indicator = np.asarray(indicator).T  # shape(2/4/8, -1)
    indicator = np.concatenate((indicator[:,0::2], indicator[:, 1::2]), axis=1)  # (real, imag)
    return indicator


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))  # RMS of OFDM signal
    CL = CL*sigma   # clipping level
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
    #print (sum(abs(x_clipped_temp-x_clipped)))
    return x_clipped


def PAPR(x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB


def QAM_Modulation(bits, mu, modem=None):
    if mu == 1:
        bits_mod = (2*bits-1).reshape(int(len(bits)),1)
    elif mu == 2:
        bits_mod = Modulation(bits)/sq2
    elif mu == 4:
        bits_mod = Modulation_16(bits)/sq10
    elif mu == 6:
        bits_mod = Modulation_64(bits)/sq42
    else:  # for high-order modulation
        bits_mod = modem.modulate(bits) / np.sqrt(modem.Es)
    return bits_mod


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/2), 2))  # real & imag
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)  # This is just for QAM modulation
#    return np.concatenate((2*bit_r[:,0]-1, 2*bit_r[:,1]-1))


# mapping
def Modulation_16(bits):
    bit_r = bits.reshape((int(len(bits)/4), 4))
    bit_mod = []
    for i in range(int(len(bits)/4)):
        bit_mod.append(list(_16QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def Modulation_64(bits):
    bit_r = bits.reshape((int(len(bits)/6), 6))
    bit_mod = []
    for i in range(int(len(bits)/6)):
        bit_mod.append(list(_64QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP == 0:
        return OFDM_time
    elif CP_flag is False:
        # add noise CP——no ISI, only ICI
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        codeword_noise = QAM_Modulation(bits_noise,mu)
        OFDM_time_noise = np.fft.ifft(codeword_noise)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape) +
                                 1j*np.random.randn(*convolved.shape))
    return convolved + noise,sigma2


def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]    # cp~cp+K


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def QAM_Demodulation(bits_mod, mu, modem=None):
    if mu == 1:
        bits_demod = abs(bits_mod+1) >= abs(bits_mod-1)
        bits_demod = bits_demod.astype(np.int32).reshape(-1)
    elif mu == 2:
        bits_demod = Demodulation(bits_mod*sq2)
    elif mu == 4:
        bits_demod = Demodulation_16(bits_mod*sq10)
    elif mu == 6:
        bits_demod = Demodulation_64(bits_mod*sq42)
    else:
        bits_demod = modem.demodulate(bits_mod.reshape(-1) * np.sqrt(modem.Es), demod_type='hard')
    return bits_demod


def Demodulation(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((4,1))
        min_distance_index = np.argmin(abs(tmp - _QPSK_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_QPSK_demapping_table[
            tuple(_QPSK_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_16(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((16,1))
        min_distance_index = np.argmin(abs(tmp - _16QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_16QAM_demapping_table[
            tuple(_16QAM_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_64(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((64,1))
        min_distance_index = np.argmin(abs(tmp - _64QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_64QAM_demapping_table[
            tuple(_64QAM_Constellation[min_distance_index])])))
    return X_pred


def PS(bits):
    return bits.reshape((-1,))


def NLE(vle,ule,orth=True,mu=2,SE=False,x=None,EP=False,soft=False,norm=1):
    if soft:
        ext_probs = np.zeros((ule.shape[0], 2**(mu//2)))
    # for QPSK signal
    if mu == 2:  # {-1,+1}
        P0 = np.maximum(np.exp(-(-1/sq2/norm-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq2/norm-ule)**2/(2*vle)),1e-100)
        u_post = (P1-P0) / (P1+P0)/sq2/norm
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P0*(u_post+1/sq2/norm)**2+P1*(u_post-1/sq2/norm)**2)/(P1+P0)
        if soft:
            ext_probs[:, 0], ext_probs[:, 1] = P0.reshape(-1), P1.reshape(-1)
    elif mu == 4:  # {-3,-1,+1,+3}
        P_3 = np.maximum(np.exp(-(-3/sq10/norm-ule)**2/(2*vle)),1e-100)
        P_1 = np.maximum(np.exp(-(-1/sq10/norm-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq10/norm-ule)**2/(2*vle)),1e-100)
        P3 = np.maximum(np.exp(-(3/sq10/norm-ule)**2/(2*vle)),1e-100)
        u_post = (-3*P_3-P_1+P1+3*P3) / (P_3+P_1+P1+P3)/sq10/norm
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P_3*(u_post+3/sq10/norm)**2+P_1*(u_post+1/sq10/norm)**2 +
                      P1*(u_post-1/sq10/norm)**2+P3*(u_post-3/sq10/norm)**2)/(P_3+P_1+P1+P3)
        if soft:
            ext_probs[:, 0], ext_probs[:, 1] = P_3.reshape(-1), P_1.reshape(-1)
            ext_probs[:, 2], ext_probs[:, 3] = P3.reshape(-1), P1.reshape(-1)
    else:  # {-7,-5,-3,-1,+1,+3,+5,+7}
        P_7 = np.maximum(np.exp(-(-7/sq42/norm-ule)**2/(2*vle)),1e-100)
        P_5 = np.maximum(np.exp(-(-5/sq42/norm-ule)**2/(2*vle)),1e-100)
        P_3 = np.maximum(np.exp(-(-3/sq42/norm-ule)**2/(2*vle)),1e-100)
        P_1 = np.maximum(np.exp(-(-1/sq42/norm-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq42/norm-ule)**2/(2*vle)),1e-100)
        P3 = np.maximum(np.exp(-(3/sq42/norm-ule)**2/(2*vle)),1e-100)
        P5 = np.maximum(np.exp(-(5/sq42/norm-ule)**2/(2*vle)),1e-100)
        P7 = np.maximum(np.exp(-(7/sq42/norm-ule)**2/(2*vle)),1e-100)
        u_post = (-7*P_7-5*P_5-3*P_3-P_1+P1+3*P3+5*P5+7*P7) / \
                 (P_7+P_5+P_3+P_1+P1+P3+P5+P7)/sq42/norm
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P_7*(u_post+7/sq42/norm)**2+P_5*(u_post+5/sq42/norm)**2 +
                      P_3*(u_post+3/sq42/norm)**2+P_1*(u_post+1/sq42/norm)**2 +
                      P1*(u_post-1/sq42/norm)**2+P3*(u_post-3/sq42/norm)**2 +
                      P5*(u_post-5/sq42/norm)**2+P7*(u_post-7/sq42/norm)**2) / \
                     (P_7+P_5+P_3+P_1+P1+P3+P5+P7)
        if soft:
            ext_probs[:, 0], ext_probs[:, 1] = P_7.reshape(-1), P_5.reshape(-1)
            ext_probs[:, 2], ext_probs[:, 3] = P_1.reshape(-1), P_3.reshape(-1)
            ext_probs[:, 4], ext_probs[:, 5] = P7.reshape(-1), P5.reshape(-1)
            ext_probs[:, 6], ext_probs[:, 7] = P1.reshape(-1), P3.reshape(-1)
    if EP is False:
        v_post = np.mean(v_post)

    if orth:
        u_orth = (u_post/v_post-ule/vle)/(1/v_post-1/vle)
        v_orth = 1/(1/v_post-1/vle)
    else:
        u_orth = u_post
        v_orth = v_post

    if soft:
        return u_post,v_post,u_orth,v_orth,ext_probs
    return u_post,v_post,u_orth,v_orth


def de2bi(decimal, order):  # decimal to binary
    binary = np.zeros((len(decimal), order), dtype = int)
    for i in range(len(decimal)):
        temp = bin(decimal[i])[2:]  # remove '0b'
        for j in range(order - len(temp)):
            binary[i, j] = 0
        for j in range(len(temp)):
            binary[i, order - len(temp) + j] = temp[j]
    return binary


def conv_encoder(in_bits):
    gen_poly = np.array([[1,0,1,1,0,1,1],[1,1,1,1,0,0,1]])
    row = gen_poly.shape[0]
    num_of_bits = gen_poly.shape[1] + len(in_bits) - 1

    coded_bits = np.zeros((row,num_of_bits), dtype=int)
    for r in range(row):
        coded_bits[r,:] = np.convolve(in_bits, gen_poly[r,:]) % 2

    return coded_bits


def viterbi_init():
    gen_poly = np.array([[1,0,1,1,0,1,1],[1,1,1,1,0,0,1]])  # memory size = 6-->state = 64
    global prev_state  # declare the variable as global
    global prev_state_outbits
    prev_state = np.zeros((64,2),dtype=int)
    prev_state_outbits = np.zeros((64,2,2),dtype=int)

    for state in range(64):
        state_bits = np.zeros(6, dtype=int)
        for i in range(2,len(bin(state))):
            state_bits[6-i+1] = int(bin(state)[i])
        if len(bin(state))-2 < 6:
            state_bits = np.roll(state_bits,len(bin(state))-2)
        input_bit = state_bits[0]
        for transition in range(2):
            prev_state_bits = np.append(state_bits[1:],transition)
            temp = ''
            for i in range(6):
                temp += str(prev_state_bits[6-1-i])
            prev_state[state,transition] = int(temp, base=2)
            prev_state_outbits[state,transition,0] = 2*(sum(
                        gen_poly[0,:]*np.append(input_bit,prev_state_bits)) % 2)-1
            prev_state_outbits[state,transition,1] = 2*(sum(
                        gen_poly[1,:]*np.append(input_bit,prev_state_bits)) % 2)-1


def viterbi_decode(rx_bits):
    global prev_state  # declare the variable as global
    global prev_state_outbits

    rx_bits = 2*rx_bits-1
    cum_metrics = -1e-6 * np.ones((64,1))
    cum_metrics[0] = 0

    tmp_cum_metrics = np.zeros((64,1))
    max_paths = np.zeros((64,len(rx_bits)//2),dtype=int)   # Rc = 1/2
    out_bits = np.zeros(len(rx_bits)//2,dtype=int)

    for data_bit in range(0,len(rx_bits),2):
        for state in range(64):
            path_metric1 = prev_state_outbits[state,0,0]*rx_bits[data_bit] +\
                           prev_state_outbits[state,0,1]*rx_bits[data_bit+1]
            path_metric2 = prev_state_outbits[state,1,0]*rx_bits[data_bit] +\
                           prev_state_outbits[state,1,1]*rx_bits[data_bit+1]

            if cum_metrics[prev_state[state,0]] + path_metric1 >\
               cum_metrics[prev_state[state,1]] + path_metric2:
                tmp_cum_metrics[state] = cum_metrics[prev_state[state,0]] + path_metric1
                max_paths[state,(data_bit+1)//2] = 0
            else:
                tmp_cum_metrics[state] = cum_metrics[prev_state[state,1]] + path_metric2
                max_paths[state,(data_bit+1)//2] = 1

        # refresh the state
        for state in range(64):
            cum_metrics[state] = tmp_cum_metrics[state]

    # trace back
    state = 0
    for data_bit in range(len(rx_bits)//2-1,-1,-1):
        bit_estimate = state % 2
        out_bits[data_bit] = bit_estimate
        state = prev_state[state,max_paths[state,data_bit]]

    return out_bits


def viterbi_decode_soft(rx_bits):
    global prev_state  # declare the variable as global
    global prev_state_outbits

    cum_metrics = -1e-6 * np.ones((64,1))
    cum_metrics[0] = 0

    tmp_cum_metrics = np.zeros((64,1))
    max_paths = np.zeros((64,len(rx_bits)//2),dtype=int)   # Rc = 1/2
    out_bits = np.zeros(len(rx_bits)//2,dtype=int)

    for data_bit in range(0,len(rx_bits),2):
        for state in range(64):
            path_metric1 = prev_state_outbits[state,0,0]*rx_bits[data_bit] +\
                        prev_state_outbits[state,0,1]*rx_bits[data_bit+1]
            path_metric2 = prev_state_outbits[state,1,0]*rx_bits[data_bit] +\
                        prev_state_outbits[state,1,1]*rx_bits[data_bit+1]

            if cum_metrics[prev_state[state,0]] + path_metric1 >\
                cum_metrics[prev_state[state,1]] + path_metric2:
                tmp_cum_metrics[state] = cum_metrics[prev_state[state,0]] + path_metric1
                max_paths[state,(data_bit+1)//2] = 0
            else:
                tmp_cum_metrics[state] = cum_metrics[prev_state[state,1]] + path_metric2
                max_paths[state,(data_bit+1)//2] = 1

        # refresh the state
        for state in range(64):
            cum_metrics[state] = tmp_cum_metrics[state]

    # trace back
    state = 0
    for data_bit in range(len(rx_bits)//2-1,-1,-1):
        bit_estimate = state % 2
        out_bits[data_bit] = bit_estimate
        state = prev_state[state,max_paths[state,data_bit]]

    return out_bits


def bcjr(trellis, llr_a2):  # todo: debug
    # initialization
    # inf = 1e+100
    numInputSymbols, numOutputSymbols = trellis.number_inputs, 2**trellis.n
    numInputBits, numOutputBits = int(np.log2(numInputSymbols)), int(np.log2(numOutputSymbols))
    numStates = trellis.number_states
    nextStatesArr = trellis.next_state_table
    outSymbolsArr = trellis.output_table
    numBits = len(llr_a2)
    numSymbols = numBits // numOutputBits
    # detected systematic message
    xhatk0 = np.zeros(numInputBits*numSymbols)
    xhatk1 = np.zeros(numInputBits*numSymbols)
    xhatk = np.zeros(numInputBits * numSymbols, dtype=int)
    # a posteriori llr output
    llr_0 = np.zeros(numBits)
    llr_1 = np.zeros(numBits)
    llr_d = np.zeros(numBits)
    # Branch Metric Vector (precomputed)
    bmVector = np.zeros((numOutputSymbols, 1))

    # Alpha-Metric (Forward Iteration)
    alphaMetric = -np.inf *np.ones((numStates, numSymbols+1))  # todo: -DBL_MAX & why +1 & [i,j] = -DBL_MAX
    # for i in range(numSymbols):
    #     for j in range(numStates):
    #         alphaMetric[j, i] = -np.inf
    alphaMetric[0, 0] = 100000.0  # start in state 0, severely discriminate against all other states

    for i in range(numSymbols):
        # precompute the Branch Metric for each output Symbol, based on the LLRs
        for q in range(numOutputSymbols):  # todo: left-msb or right-msb ?
            bmVector[q] = 0.  # !!
            tmp = q
            # for b in range(numOutputBits, 0, -1):
            #     if tmp >= 2**(b-1):
            #         tmp = tmp - 2**(b-1)
            #         bmVector[q] += llr_a2[i*numOutputBits+numOutputBits-b]  # first bit is MSB
            #     else:
            #         bmVector[q] -= llr_a2[i*numOutputBits+numOutputBits-b]
            for b in range(numOutputBits-1, -1, -1):
                if tmp >= (1<<b):
                    tmp = tmp - (1<<b)
                    bmVector[q] += llr_a2[i*numOutputBits+numOutputBits-b-1]  # first bit is MSB
                else:
                    bmVector[q] -= llr_a2[i*numOutputBits+numOutputBits-b-1]
            # adjust to max-log metric
            bmVector[q] = 0.5 * bmVector[q]

        # carry out one trellis step
        for j in range(numStates):  # for each state
            prevAlpha = alphaMetric[j, i]
            # for each branch
            for p in range(numInputSymbols):
                curMetric = prevAlpha + bmVector[outSymbolsArr[j,p]]
                nextState = nextStatesArr[j, p]
                if curMetric > alphaMetric[nextState, i+1]:  # max-log
                    alphaMetric[nextState, i+1] = curMetric
                # alphaMetric[nextState, i + 1] = jacobilog(curMetric, alphaMetric[nextState, i + 1])  # log-MAP

    # Beta-Metric (Backward Iteration)
    betaMetric = np.zeros((numStates, 1))
    # oldBetaMetric = np.zeros((numStates, 1))
    oldBetaMetric = -np.inf * np.ones((numStates, 1))
    oldBetaMetric[0, 0] = 100000.0  # termination

    for i in range(numSymbols, 0, -1):
        # precompute the Branch Metric for each output Symbol, based on the LLRs
        for q in range(numOutputSymbols):
            bmVector[q] = 0.
            tmp = q
            # for b in range(numOutputBits, 0, -1):
            #     if tmp >= 2**(b-1):
            #         tmp = tmp - 2**(b-1)
            #         bmVector[q] += llr_a2[i*numOutputBits-b]
            #     else:
            #         bmVector[q] -= llr_a2[i*numOutputBits-b]
            for b in range(numOutputBits-1, -1, -1):
                if tmp >= (1<<b):
                    tmp = tmp - (1<<b)
                    bmVector[q] += llr_a2[i*numOutputBits-b-1]
                else:
                    bmVector[q] -= llr_a2[i*numOutputBits-b-1]
            # adjust to max-log metric
            bmVector[q] = 0.5 * bmVector[q]

        # carry out one trellis step
        for j in range(numStates):  # for each state
            # for each branch
            for p in range(numInputSymbols):
                metricInc = bmVector[outSymbolsArr[j,p]]
                nextState = nextStatesArr[j,p]
                curMetric = oldBetaMetric[nextState] + metricInc
                if curMetric > betaMetric[j] or p == 0:  # max-log
                    betaMetric[j] = curMetric
                # if p == 0:
                #     betaMetric[j] = curMetric
                # else:
                #     betaMetric[j] = jacobilog(curMetric, betaMetric[j])  # log-map

                # output metric of current state
                outputMetric = curMetric + alphaMetric[j, i-1]
                # LLR output of input bits
                for b in range(numInputBits-1, -1, -1):
                    bitIdx = numInputBits*i - 1 - b
                    if p & (1<<b):  # bit is +1
                        if outputMetric > xhatk1[bitIdx]:   # max-log
                            xhatk1[bitIdx] = outputMetric
                        # xhatk1[bitIdx] = jacobilog(outputMetric, xhatk1[bitIdx])  # log-map
                    else:
                        if outputMetric > xhatk0[bitIdx]:
                            xhatk0[bitIdx] = outputMetric
                        # xhatk0[bitIdx] = jacobilog(outputMetric, xhatk0[bitIdx])  # log-map

                # a posteriori LLR output calculation
                bit = outSymbolsArr[j,p]
                for b in range(numOutputBits-1, -1, -1):
                    bitIdx = numOutputBits * i - 1 - b
                    if bit >= (1<<b):  # bit is +1
                        bit = bit - (1<<b)
                # for b in range(numOutputBits, 0, -1):
                #     bitIdx = numOutputBits * i - b
                #     if bit >= 2**(b-1):  # bit is +1
                #         bit = bit - 2**(b-1)
                        if outputMetric>llr_1[bitIdx]:
                            llr_1[bitIdx] = outputMetric
                        # llr_1[bitIdx] = jacobilog(outputMetric, llr_1[bitIdx])
                    else:  # bit is -1
                        if outputMetric>llr_0[bitIdx]:
                            llr_0[bitIdx] = outputMetric
                        # llr_0[bitIdx] = jacobilog(outputMetric, llr_0[bitIdx])

        # copy the beta state metric
        for q in range(numStates):
            oldBetaMetric[q] = betaMetric[q]

    # compute the sliced systematic output bits
    for i in range(numInputBits*numSymbols):
        if xhatk1[i] > xhatk0[i]:
            xhatk[i] = 1
        else:
            xhatk[i] = 0

    # compute a posterior information [LLRs]
    for i in range(numBits):
        llr_d[i] = llr_1[i] - llr_0[i]

    return llr_d, xhatk


def calc_llr(N, mu, ext_probs):
    llr_d = np.zeros((N, mu))
    bin_array = np.sign(de2bi(np.arange(2 ** mu), mu) - 0.5).astype(np.int)  # (2 ** mu, mu)
    eps = 1e-16
    for n in range(N):
        for b in range(mu):
            pos, neg = 0., 0.
            for z in range(2 ** mu):
                if bin_array[z, b] == 1:
                    pos += ext_probs[n, z]
                else:
                    neg += ext_probs[n, z]
            llr_d[n, b] = np.log(pos + eps) - np.log(neg + eps)
    # output extrinsic information
    llr_e = llr_d
    return llr_e


def calc_llr_real(N, mu, ext_probs):  # N, ext_probs both have size enlarged by 2
    llr_real = np.zeros((N, mu // 2))
    bin_array = de2bi(np.arange(2 ** (mu // 2)), mu // 2)
    eps = 1e-15
    for n in range(N):
        for b in range(mu // 2):
            pos, neg = 0., 0.
            for z in range(2 ** (mu // 2)):
                if bin_array[z, b] == 1:
                    pos += ext_probs[n, z]
                else:
                    neg += ext_probs[n, z]
            llr_real[n, b] = np.log(pos + eps) - np.log(neg + eps)
    llr = np.zeros((N//2, mu))
    llr[:, :mu // 2], llr[:, mu // 2:] = llr_real[:N//2, :], llr_real[N//2:, :]
    return llr


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def puncturing(code_rate, total_coded_bits, trellis):
    if code_rate == 5/6:
        period = 10
        pattern = [1, 2, 3, 6, 7, 10]
        pat_len = len(pattern)
        index = []
        for i in range(int(np.ceil(total_coded_bits / pat_len))):
            for j in range(pat_len):
                index.append(i * period + pattern[j] - 1)
        trellis.puncturning_index = np.array(index)
    return trellis


def lmmse_ce(xp_mat, yp_mat, sigma2, rhh):
    Nr, Np = xp_mat.shape
    A = np.kron(np.eye(Nr), xp_mat.T)
    AH = np.conj(A.T)
    wlmmse = rhh @ AH @ la.inv(A @ rhh @ AH + sigma2 * np.eye(Nr * Np))
    return wlmmse