#!/usr/bin/python
from __future__ import division
import numpy as np
import math
import os
import time
import sys
import scipy.interpolate
import numpy.linalg as la
import tensorflow as tf
#from .tfinterp import interp1d_
# np.random.seed(1) # numpy is good about making repeatable output

sqrt=np.sqrt
pi = math.pi

K = 64
CP = K//4   #‘//’向下取整除，CP=16
P = 64 # number of pilot carriers per OFDM block
#pilotValue = 1+1j
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.第一帧插入导频的位置(只在P<K时关注)
#pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
#P = P+1
dataCarriers = np.delete(allCarriers, pilotCarriers)  #对K=P的情景不适用吗？
mu = 4
#payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

payloadBits_per_OFDM = K*mu   #把所有bit都算作payload，针对第二帧？64*2=128bit/frame
CR = 1 

SNRdb = 25  # signal to noise-ratio in dB at the receiver
Clipping_Flag = False
CP_flag = False

_QPSK_mapping_table = {
    (0,1) : (-1+1j,), (1,1) : (1+1j,),
    (0,0) : (-1-1j,), (1,0) : (1-1j,)
}

_QPSK_demapping_table = {v : k for k, v in _QPSK_mapping_table.items()}

_QPSK_Constellation = np.array([[-1+1j], [1+1j],
                                  [-1-1j], [1-1j]
                                  ])

_16QAM_mapping_table = {
    (0,0,1,0) : (-3+3j,), (0,1,1,0) : (-1+3j,), (1,1,1,0) : (1+3j,), (1,0,1,0) : (3+3j,),
    (0,0,1,1) : (-3+1j,), (0,1,1,1) : (-1+1j,), (1,1,1,1) : (1+1j,), (1,0,1,1) : (3+1j,),
    (0,0,0,1) : (-3-1j,), (0,1,0,1) : (-1-1j,), (1,1,0,1) : (1-1j,), (1,0,0,1) : (3-1j,),
    (0,0,0,0) : (-3-3j,), (0,1,0,0) : (-1-3j,), (1,1,0,0) : (1-3j,), (1,0,0,0) : (3-3j,)
}

_16QAM_demapping_table = {v : k for k, v in _16QAM_mapping_table.items()}

_16QAM_Constellation = np.array([[-3+3j], [-1+3j], [1+3j], [3+3j],
                                  [-3+1j], [-1+1j], [1+1j], [3+1j],
                                  [-3-1j], [-1-1j], [1-1j], [3-1j],
                                  [-3-3j], [-1-3j], [1-3j], [3-3j]
                                  ])

_64QAM_mapping_table = {
    (0,0,0,1,0,0) : (-7+7j,), (0,0,1,1,0,0) : (-5+7j,), (0,1,1,1,0,0) : (-3+7j,), (0,1,0,1,0,0) : (-1+7j,), (1,1,0,1,0,0) : (1+7j,), (1,1,1,1,0,0) : (3+7j,), (1,0,1,1,0,0) : (5+7j,), (1,0,0,1,0,0) : (7+7j,),
    (0,0,0,1,0,1) : (-7+5j,), (0,0,1,1,0,1) : (-5+5j,), (0,1,1,1,0,1) : (-3+5j,), (0,1,0,1,0,1) : (-1+5j,), (1,1,0,1,0,1) : (1+5j,), (1,1,1,1,0,1) : (3+5j,), (1,0,1,1,0,1) : (5+5j,), (1,0,0,1,0,1) : (7+5j,),
    (0,0,0,1,1,1) : (-7+3j,), (0,0,1,1,1,1) : (-5+3j,), (0,1,1,1,1,1) : (-3+3j,), (0,1,0,1,1,1) : (-1+3j,), (1,1,0,1,1,1) : (1+3j,), (1,1,1,1,1,1) : (3+3j,), (1,0,1,1,1,1) : (5+3j,), (1,0,0,1,1,1) : (7+3j,),
    (0,0,0,1,1,0) : (-7+1j,), (0,0,1,1,1,0) : (-5+1j,), (0,1,1,1,1,0) : (-3+1j,), (0,1,0,1,1,0) : (-1+1j,), (1,1,0,1,1,0) : (1+1j,), (1,1,1,1,1,0) : (3+1j,), (1,0,1,1,1,0) : (5+1j,), (1,0,0,1,1,0) : (7+1j,),
    (0,0,0,0,1,0) : (-7-1j,), (0,0,1,0,1,0) : (-5-1j,), (0,1,1,0,1,0) : (-3-1j,), (0,1,0,0,1,0) : (-1-1j,), (1,1,0,0,1,0) : (1-1j,), (1,1,1,0,1,0) : (3-1j,), (1,0,1,0,1,0) : (5-1j,), (1,0,0,0,1,0) : (7-1j,),
    (0,0,0,0,1,1) : (-7-3j,), (0,0,1,0,1,1) : (-5-3j,), (0,1,1,0,1,1) : (-3-3j,), (0,1,0,0,1,1) : (-1-3j,), (1,1,0,0,1,1) : (1-3j,), (1,1,1,0,1,1) : (3-3j,), (1,0,1,0,1,1) : (5-3j,), (1,0,0,0,1,1) : (7-3j,),
    (0,0,0,0,0,1) : (-7-5j,), (0,0,1,0,0,1) : (-5-5j,), (0,1,1,0,0,1) : (-3-5j,), (0,1,0,0,0,1) : (-1-5j,), (1,1,0,0,0,1) : (1-5j,), (1,1,1,0,0,1) : (3-5j,), (1,0,1,0,0,1) : (5-5j,), (1,0,0,0,0,1) : (7-5j,),
    (0,0,0,0,0,0) : (-7-7j,), (0,0,1,0,0,0) : (-5-7j,), (0,1,1,0,0,0) : (-3-7j,), (0,1,0,0,0,0) : (-1-7j,), (1,1,0,0,0,0) : (1-7j,), (1,1,1,0,0,0) : (3-7j,), (1,0,1,0,0,0) : (5-7j,), (1,0,0,0,0,0) : (7-7j,)
}

_64QAM_demapping_table = {v : k for k, v in _64QAM_mapping_table.items()}

_64QAM_Constellation = np.array([[-7+7j], [-5+7j], [-3+7j], [-1+7j], [1+7j], [3+7j], [5+7j], [7+7j],
                                [-7+5j], [-5+5j], [-3+5j], [-1+5j], [1+5j], [3+5j], [5+5j], [7+5j],
                                [-7+3j], [-5+3j], [-3+3j], [-1+3j], [1+3j], [3+3j], [5+3j], [7+3j],
                                [-7+1j], [-5+1j], [-3+1j], [-1+1j], [1+1j], [3+1j], [5+1j], [7+1j],
                                [-7-1j], [-5-1j], [-3-1j], [-1-1j], [1-1j], [3-1j], [5-1j], [7-1j],
                                [-7-3j], [-5-3j], [-3-3j], [-1-3j], [1-3j], [3-3j], [5-3j], [7-3j],
                                [-7-5j], [-5-5j], [-3-5j], [-1-5j], [1-5j], [3-5j], [5-5j], [7-5j],
                                [-7-7j], [-5-7j], [-3-7j], [-1-7j], [1-7j], [3-7j], [5-7j], [7-7j]
                                  ])


def Clipping (x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))  #OFDM信号的RMS
    CL = CL*sigma   #限幅电平
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))   #公式参考MIMO-OFDM
    #print (sum(abs(x_clipped_temp-x_clipped)))
    return x_clipped

def PAPR (x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB

def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/2), 2)) #real & imag
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)    # This is just for QAM modulation，m=4？ 0->-1 1->+1
#    return np.concatenate((2*bit_r[:,0]-1, 2*bit_r[:,1]-1))
#mapping实现  其他方法：卡诺图实现
def Modulation_16(bits):
    bit_r = bits.reshape((int(len(bits)/4), 4))
    bit_mod = []
    for i in range(int(len(bits)/4)):
        bit_mod.append( list( _16QAM_mapping_table.get( tuple(bit_r[i]) ) ) )
    return np.asarray(bit_mod).reshape((-1,))

def Modulation_64(bits):
    bit_r = bits.reshape((int(len(bits)/6), 6))
    bit_mod = []
    for i in range(int(len(bits)/6)):
        bit_mod.append( list( _64QAM_mapping_table.get( tuple(bit_r[i]) ) ) )
    return np.asarray(bit_mod).reshape((-1,))

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time, CP, CP_flag, mu, K):    
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        codeword_noise = Modulation_64(bits_noise)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2)) #功率为平方的均值
    sigma2 = signal_power * 10**(-SNRdb/10) #由信噪比和信号功率反推噪声功率
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape)) #噪声幅度/均值+高斯随机变量？
    return convolved + noise,sigma2

def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]    #cp~cp+K

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def Demodulation(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((4,1))
        min_distance_index = np.argmin(abs(tmp - _QPSK_Constellation))
        X_pred = np.concatenate((X_pred,np.array( _QPSK_demapping_table[ tuple(_QPSK_Constellation[min_distance_index]) ])))
    return X_pred

def Demodulation_16(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((16,1))
        min_distance_index = np.argmin(abs(tmp - _16QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array( _16QAM_demapping_table[ tuple(_16QAM_Constellation[min_distance_index]) ])))
    return X_pred

def Demodulation_64(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((64,1))
        min_distance_index = np.argmin(abs(tmp - _64QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array( _64QAM_demapping_table[ tuple(_64QAM_Constellation[min_distance_index]) ])))
    return X_pred

def get_payload(equalized):
    return equalized[dataCarriers]

def PS(bits):
    return bits.reshape((-1,))

def ofdm_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers) #with different pilots
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation_64(bits)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    
    OFDM_time = IDFT(OFDM_data) #IDFT变换到时域
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K) #插入CP
    OFDM_TX = OFDM_withCP   #发送信号
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX,CR)  # add clipping                            # add clipping 
    OFDM_RX,_ = channel(OFDM_TX, channelResponse,SNRdb)   #通过信道后输出
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,K)  #去除CP

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation_64(codeword)
    if len(codeword_qam) != K:
        print ('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol #步骤同上
    OFDM_time_codeword = IDFT(OFDM_data_codeword)
    OFDM_withCP_codeword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    if Clipping_Flag:
        OFDM_withCP_codeword = Clipping(OFDM_withCP_codeword,CR) # add clipping 
    OFDM_RX_codeword,sigma2 = channel(OFDM_withCP_codeword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), sigma2 #sparse_mask

def ofdm_simulate_cp_free(codeword, H, A, FH, SNR, mu, K, P, pilotValue,pilotCarriers, dataCarriers, CE_flag=False):
    payloadBits_per_OFDM = mu*len(dataCarriers) #with different pilots
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if mu == 2:
            QAM = Modulation(bits)
        elif mu == 4:
            QAM = Modulation_16(bits)
        else:     
            QAM = Modulation_64(bits)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue  #up
    yp = (H-A) @ FH @ OFDM_data
    signal_power = np.mean(abs(yp**2))
    sigma2 = signal_power * 10**(-SNR/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*yp.shape)+1j*np.random.randn(*yp.shape))
    yp = yp + noise
    #add ISI
    # bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))  #previous??
    # if mu == 2:
    #     codeword_noise = Modulation(bits_noise)
    # elif mu == 4:
    #     codeword_noise = Modulation_16(bits_noise)
    # else:     
    #     codeword_noise = Modulation_64(bits_noise)
    # OFDM_data_nosie = codeword_noise
    # ISI = A @ FH @ OFDM_data_nosie
    # yp = yp+ISI

    if CE_flag:
        return np.concatenate((np.real(yp),np.imag(yp)))
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    if mu == 2:
        codeword_qam = Modulation(codeword)
    elif mu == 4:
        codeword_qam = Modulation_16(codeword)
    else:     
        codeword_qam = Modulation_64(codeword)
    if len(codeword_qam) != K:
        print ('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol #步骤同上
    ys = (H-A) @ FH @ OFDM_data_codeword
    signal_power = np.mean(abs(ys**2))
    sigma2 = signal_power * 10**(-SNR/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*ys.shape)+1j*np.random.randn(*ys.shape))
    ys = ys + noise
    #add ISI
    # bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))  #previous??
    # if mu == 2:
    #     codeword_noise = Modulation(bits_noise)
    # elif mu == 4:
    #     codeword_noise = Modulation_16(bits_noise)
    # else:     
    #     codeword_noise = Modulation_64(bits_noise)
    # OFDM_data_nosie = codeword_noise
    # ISI = A @ FH @ OFDM_data_nosie
    # ys = ys+ISI

    return np.concatenate((np.concatenate((np.real(yp),np.imag(yp))), np.concatenate((np.real(ys),np.imag(ys))))),\
        sigma2, codeword_qam 

def LS_CE(Y,pilotValue,pilotCarriers,K,P,int_opt):
    index = np.arange(P)
    LS_est = np.zeros(P, dtype=complex)
    LS_est[index] = Y[pilotCarriers] / pilotValue[index]    #complex
    if int_opt == 0:
        H_LS = interpolate(LS_est,pilotCarriers,K,0)
    if int_opt == 1:
        H_LS = interpolate(LS_est,pilotCarriers,K,1)
    return H_LS

def MMSE_CE(Y,pilotValue,pilotCarriers,K,P,h,SNR):
    snr = 10 ** (SNR*0.1)
    index = np.arange(P)
    H_tilde = np.zeros(P, dtype=complex)
    H_tilde[index] = Y[pilotCarriers] / pilotValue[index] #LS estimation
    index = np.arange(len(h))
    hh = h.dot(np.conj(h).T) #厄密共轭！！
    tmp = h *np.conj(h)*index
    r = np.sum(tmp)/hh
    r2 = tmp.dot(index.T) / hh
    tau_rms = (r2 - r**2)**0.5
    df = 1 / K
    j2pi_tau_df = 1j*2*math.pi*tau_rms*df
    K1 = np.reshape(np.repeat(np.arange(K).T, P),(K,P))
    K2 = np.arange(P)
    for i in range(K-1):
        K2 = np.concatenate((K2,np.arange(P)))
    K2 = np.reshape(K2,(K,P))
    rf = np.ones((K,P),dtype=complex) / (1+j2pi_tau_df*(K1-K2*(K//P)))
    K3 = np.reshape(np.repeat(np.arange(P).T, P),(P,P))
    K4 = np.arange(P)
    for i in range(P-1):
        K4 = np.concatenate((K4,np.arange(P)))
    K4 = np.reshape(K4,(P,P))
    rf2 = np.ones((P,P),dtype=complex) / (1+j2pi_tau_df*(K//P)*(K3-K4))
    Rhp = rf
    Rpp = rf2 + np.eye(len(H_tilde)) / snr
    W_MMSE = Rhp.dot(np.linalg.inv(Rpp))
    H_MMSE = (W_MMSE.dot(H_tilde.T)).T

    return H_MMSE,W_MMSE

interpolate_method = 1
def interpolate(H_est,pilotCarriers,K,method): #for P<K 只能在内部插  complex can be interpolate?
    if pilotCarriers[0] > 0 :
        slope = (H_est[1]-H_est[0])/(K//P)
        H_est = np.insert(H_est,0, H_est[0]-slope*(K//P))
        pilotCarriers = np.insert(pilotCarriers,0,0)
    if pilotCarriers[len(pilotCarriers) - 1] < (K-1):
        slope = (H_est[len(H_est)-1]-H_est[len(H_est)-2])/(K//P)
        H_est = np.append(H_est,H_est[len(H_est)-1]+slope*(K//P))
        pilotCarriers = np.append(pilotCarriers,(K-1))
    if method == 0:
        H_interpolated = scipy.interpolate.interp1d(pilotCarriers, H_est,'linear')  #线性插值
    if method == 1:
        H_interpolated = scipy.interpolate.interp1d(pilotCarriers, H_est,'cubic')   #三次样条插值
    index = np.arange(K)
    H_interpolated_new = H_interpolated(index)
    return H_interpolated_new

def Normalized_FFT_Matrix(K):
    F = np.zeros((K,K),dtype=complex)
    for i in range(K):
        for j in range(K):
             F[i][j]=1/np.sqrt(K)*np.exp(-1j*2*pi*i*j/K)
    return F

# rho = 0.5
# R = np.zeros((Nt,Nt))   #存储相关矩阵
# for i in range(R.shape[0]):#生成相关矩阵
#     for j in range(R.shape[1]):
#         if i <= j:
#             R[i][j] = rho**(j-i)
#         else:
#             R[i][j] = rho**(i-j)
# sqrtR = np.sqrt(R)
# H = np.sqrt(1/2)*(np.random.randn(Mr,Nt) + 1j * np.random.randn(Mr,Nt)) #i.i.d. Rayleigh channel
# H = np.matmul(np.matmul(sqrtR,H),sqrtR)
# print(sqrtR)

def OAMP(K,yd,H,sigma2,channel_type=0,mu=2,Mr=4,Nt=4,T=4): #用tf或numpy实现
    #initialize
    v_sqr_last = 0.
    x_hat = np.zeros((2*K,1))
    for t in range(T): #公式中sigma2应当是噪声方差noise_var
        v_sqr = (np.square(np.linalg.norm(yd-H.dot(x_hat),2,axis=0)) - K*sigma2) / np.trace(H.T.dot(H)) #(8)
        v_sqr = 0.5*v_sqr + 0.5 *v_sqr_last
        v_sqr = np.maximum(v_sqr,1e-9) # in case that v_sqr is negative
        v_sqr_last = v_sqr
        w_hat = v_sqr * H.T.dot( np.linalg.inv(v_sqr*H.dot(H.T)+sigma2/2*np.eye(2*K)) )#(11)
        w = 2*K/np.trace(w_hat.dot(H))*w_hat   #(10)
        r = x_hat + w.dot(yd-H.dot(x_hat))   #(6) denoiser input
        B = np.eye(2*K) - w.dot(H)
        tau_sqr = 1/(2*K)*np.trace(B.dot(B.T))*v_sqr + 1/(4*K)*np.trace(w.dot(w.T))*sigma2 #(9)
        tau_sqr = np.maximum(tau_sqr,1e-9)
        if mu == 2: #{-1,+1}
            P0 = np.exp(-(-1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P1 = np.exp(-(1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            x_hat = (P1-P0) / (P1+P0)   #(18)
        elif mu == 4: #{-3,-1,+1,+3}
            clipped_idx = abs(r) > 4.
            r[clipped_idx] = np.divide((r[clipped_idx]*4.),abs(r[clipped_idx]))
            P_3 = np.exp(-(-3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_1 = np.exp(-(-1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P1 = np.exp(-(1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P3 = np.exp(-(3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            x_hat = (-3*P_3-P_1+P1+3*P3) / (P_3+P_1+P1+P3)   #(18)
        else: #{-1,+1}
            #clipping
            clipped_idx = abs(r) > 8.
            r[clipped_idx] = np.divide((r[clipped_idx]*8.),abs(r[clipped_idx]))
            P_7 = np.exp(-(-7-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_5 = np.exp(-(-5-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_3 = np.exp(-(-3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_1 = np.exp(-(-1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P1 = np.exp(-(1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P3 = np.exp(-(3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P5 = np.exp(-(5-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P7 = np.exp(-(7-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            # if t==0:
            #     print(r)
            #     print(P_7+P_5+P_3+P_1+P1+P3+P5+P7)
            x_hat = (-7*P_7-5*P_5-3*P_3-P_1+P1+3*P3+5*P5+7*P7) / (P_7+P_5+P_3+P_1+P1+P3+P5+P7)   #(18)
        #ber = 1 - np.mean(np.equal(np.sign(x_hat),x).astype(int))

    # calculate the number of err bits
    #err_bits = np.sum(np.greater(abs(x_hat-x),np.ones((2*Nt,1))).astype(int)) #计算的是误码率,除了QPSK
    #back into complex
    x_hat = x_hat.reshape((2,K))
    x_hat = x_hat[0,:]+1j*x_hat[1,:]
    #demodulate
    if mu == 2:
        x_hat_demod = Demodulation(x_hat)
    elif mu == 4:
        x_hat_demod = Demodulation_16(x_hat)
    else:
        x_hat_demod = Demodulation_64(x_hat)

    return x_hat_demod

channel_train = np.load('tools/channel_train.npy')
train_size = channel_train.shape[0]  #100000
channel_test = np.load('tools/channel_test.npy')
test_size = channel_test.shape[0] #390000

def get_WMMSE(SNR):
    index = np.random.choice(np.arange(test_size), size=1)
    h = channel_test[index].reshape((-1,))
    H = np.zeros((K,K),dtype=complex)
    A = np.zeros((K,K),dtype=complex)
    F = Normalized_FFT_Matrix(K)
    FH = np.conj(F).T
    h_ = np.flip(np.append(h,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
    for i in range(K):
        H[i] = np.roll(h_,i+1)
        if i < (CP-1):
            A[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,)) #label
    signal_output = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers, CE_flag=True)
    yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
    Yp_complex = F @ yp_complex
    _,W_MMSE = MMSE_CE(Yp_complex,pilotValue,pilotCarriers,K,P,h,SNR)
    #convert complex into real
    W_MMSE = np.concatenate(( np.concatenate((np.real(W_MMSE),-np.imag(W_MMSE)),axis=1),np.concatenate((np.imag(W_MMSE),np.real(W_MMSE)),axis=1) ))
    return W_MMSE 

def sample_gen(bs, SNR = 20, training_flag=True):
    if training_flag:
        index = np.random.choice(np.arange(train_size), size=bs)    #从1*train_size的array中随机选出bs个下标
        h_total = channel_train[index]
    else:
        index = np.random.choice(np.arange(test_size), size=bs)
        h_total = channel_test[index]
    H_samples = []
    H_labels = []
    F = Normalized_FFT_Matrix(K)
    FH = np.conj(F).T
    for h in h_total:
        #labels
        H_true = np.fft.fft(h,n=K)
        H = np.zeros((K,K),dtype=complex)
        A = np.zeros((K,K),dtype=complex)
        h_ = np.flip(np.append(h,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
        for i in range(K):
            H[i] = np.roll(h_,i+1)
            if i < (CP-1):
                A[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])
        #channel estimation for the input samples
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        signal_output = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers, CE_flag=True)
        yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
        Yp_complex = F @ yp_complex
        H_LS = LS_CE(Yp_complex,pilotValue,pilotCarriers,K,P,interpolate_method) #input

        #convert complex into real
        H_true = np.concatenate((np.real(H_true),np.imag(H_true))) #1*2K
        H_LS = np.concatenate((np.real(H_LS),np.imag(H_LS)))    #1*2K

        H_labels.append(H_true)   
        H_samples.append(H_LS)
    return np.asarray(H_samples), np.asarray(H_labels)

def sample_gen_for_OAMP(bs, SNR, sess, input_holder, output, training_flag=True):
    if training_flag:
        #generate training samples:
        index = np.random.choice(np.arange(train_size), size=bs)    #从1*train_size的array中随机选出bs个下标
        h_total = channel_train[index]
    else:
        #generate development samples:
        index = np.random.choice(np.arange(test_size), size=bs)
        h_total = channel_test[index]

    H_ = np.zeros((2*bs*K,2*K),dtype=np.float32)
    x_ = np.zeros((2*bs*K,1),dtype=np.float32)
    y_ = np.zeros((2*bs*K,1),dtype=np.float32)
    sigma2_ = np.zeros((bs,1),dtype=np.float32)
    F = Normalized_FFT_Matrix(K)
    FH = np.conj(F).T
    count = 0
    for h in h_total:

        #labels
        H = np.zeros((K,K),dtype=complex)
        A = np.zeros((K,K),dtype=complex)
        h_ = np.flip(np.append(h,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
        for i in range(K):
            H[i] = np.roll(h_,i+1)
            if i < (CP-1):
                A[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])
        #channel estimation for the input samples
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        signal_output,sigma2,bits_mod = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers)
        yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
        Yp_complex = F @ yp_complex
        yd = signal_output[2*K:4*K].reshape(2*K,1)
        H_LS = LS_CE(Yp_complex,pilotValue,pilotCarriers,K,P,interpolate_method) #input

        #convert complex into real
        H_LS = np.concatenate((np.real(H_LS),np.imag(H_LS))).reshape(1,2*K)     #1*2K
        #get the CE-net's output
        H_out = sess.run(output,feed_dict={input_holder:H_LS}).reshape(-1,)
        H_est = H_out[0:K] + 1j * H_out[K:2*K]
        downsampler = allCarriers[::K//CP] #靠谱否？ 补零后FFT的逆运算为抽值后IFFT,应该是靠谱的
        H_est = H_est[downsampler]
        h_est = IDFT(H_est) #??64-->16
        H_hat = np.zeros((K,K),dtype=complex)
        A_hat = np.zeros((K,K),dtype=complex)
        h_ = np.flip(np.append(h_est,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
        for i in range(K):
            H_hat[i] = np.roll(h_,i+1)
            if i < (CP-1):
                A_hat[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])

        H_bar = (H_hat-A_hat) @ FH
        #convert complex into real
        H_bar = np.concatenate(( np.concatenate((np.real(H_bar),-np.imag(H_bar)),axis=1),\
            np.concatenate((np.imag(H_bar),np.real(H_bar)),axis=1) ))
        x = np.concatenate((np.real(bits_mod.reshape((K,1))),np.imag(bits_mod.reshape((K,1)))))
        #stack
        H_[2*K*count:2*K*(count+1)] = H_bar.astype(np.float32)
        x_[2*K*count:2*K*(count+1)] = x.astype(np.float32)
        y_[2*K*count:2*K*(count+1)] = yd.astype(np.float32)
        sigma2_[count] = sigma2.astype(np.float32)

        count = count + 1
    #reshape
    H_ = H_.reshape(bs,2*K,2*K)
    x_ = x_.reshape(bs,2*K,1)
    y_ = y_.reshape(bs,2*K,1)
    sigma2_ = sigma2_.reshape(bs,1,1)
    
    return y_,x_,H_,sigma2_

def test_DL_OAMP(sess,prob,x_hat_T,input_holder,output,SNR,OAMPnet=False):
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    start = time.time()
    while True:
        index = np.random.choice(np.arange(test_size), size=1)
        h = channel_test[index].reshape((-1,))
        H = np.zeros((K,K),dtype=complex)
        A = np.zeros((K,K),dtype=complex)
        F = Normalized_FFT_Matrix(K)
        FH = np.conj(F).T
        h_ = np.flip(np.append(h,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
        for i in range(K):
            H[i] = np.roll(h_,i+1)
            if i < (CP-1):
                A[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,)) #label
        signal_output, sigma2,_ = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers)
        yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
        Yp_complex = F @ yp_complex
        yd = signal_output[2*K:4*K].reshape(2*K,1)
        H_LS = LS_CE(Yp_complex,pilotValue,pilotCarriers,K,P,interpolate_method)
        # H_true = np.fft.fft(h,n=K)
        #convert complex into real
        H_LS = np.concatenate((np.real(H_LS),np.imag(H_LS))).reshape(1,2*K)    #1*2K
        # H_true = np.concatenate((np.real(H_true),np.imag(H_true))) #1*2K

        #get the CE-net's output
        H_out = sess.run(output,feed_dict={input_holder:H_LS}).reshape(-1,)
        #print(H_out)
        #TODO: calculate the MSE between H_out & H_true

        #convert real into complex
        H_est = H_out[0:K] + 1j * H_out[K:2*K]
        downsampler = allCarriers[::K//CP] #靠谱否？ 补零后FFT的逆运算为抽值后IFFT,应该是靠谱的
        H_est = H_est[downsampler]
        h_est = IDFT(H_est) #??64-->16
        # FH = np.conj(Normalized_FFT_Matrix(CP)).T
        # h_est = FH @ H_est
        # print("CIR:",h)
        # print("h_est:",h_est)
        H_hat = np.zeros((K,K),dtype=complex)
        A_hat = np.zeros((K,K),dtype=complex)
        h_ = np.flip(np.append(h_est,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
        for i in range(K):
            H_hat[i] = np.roll(h_,i+1)
            if i < (CP-1):
                A_hat[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])

        H_bar = (H_hat-A_hat) @ FH
        #convert complex into real
        H_bar = np.concatenate(( np.concatenate((np.real(H_bar),-np.imag(H_bar)),axis=1),np.concatenate((np.imag(H_bar),np.real(H_bar)),axis=1) ))
        if OAMPnet == False:
            #OAMP
            x_hat_demod = OAMP(K,yd,H_bar,sigma2,mu=mu)
        else:
            #get the OAMP-net's output
            yd = yd.reshape(1,2*K,1).astype(np.float32)
            H_bar = H_bar.reshape(1,2*K,2*K).astype(np.float32)
            sigma2 = sigma2.reshape(1,1,1).astype(np.float32)
            x_hat = sess.run(x_hat_T,feed_dict={prob.y_:yd,prob.x_:np.zeros((1,2*K,1),dtype=np.float32),
                prob.H_:H_bar,prob.sigma2_:sigma2,prob.sample_size_:1})
            x_hat = x_hat.reshape(2,K)
            x_hat = x_hat[0,:]+1j*x_hat[1,:]
            if mu == 2:
                x_hat_demod = Demodulation(x_hat)
            elif mu == 4:
                x_hat_demod = Demodulation_16(x_hat)
            else:
                x_hat_demod = Demodulation_64(x_hat)

        err_bits = np.sum(np.not_equal(x_hat_demod,bits))
        total_err_bits  += err_bits
        total_bits += mu*K
        if err_bits > 0:
            # print("total_err_bits:", total_err_bits,"total_bits:",total_bits,"BER:",total_err_bits/total_bits)
            sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f}'.format(teb=total_err_bits,tb=total_bits,BER=total_err_bits/total_bits))
            sys.stdout.flush()
        if total_err_bits > err_bits_target:
            end = time.time()
            print("\nSNR=",SNR,"iter_time:",end-start)
            ber = total_err_bits/total_bits
            print("BER:", ber)
            break
    return ber
  

Pilot_file_name = 'Pilot_' + str(P)+'_mu'+str(mu)+'.txt'
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(P * mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

if mu == 2:
    pilotValue = Modulation(bits)
elif mu == 4:
    pilotValue = Modulation_16(bits)
else:     
    pilotValue = Modulation_64(bits)


# BER = []
# total_time = 0
# err_bits_target = 5
# for SNR in range(40,45,5):
#     total_err_bits = 0
#     total_bits = 0
#     start = time.time()
#     while True:
#         index = np.random.choice(np.arange(test_size), size=1)
#         h = channel_test[index].reshape((-1,))
#         H = np.zeros((K,K),dtype=complex)
#         A = np.zeros((K,K),dtype=complex)
#         h_ = np.flip(np.append(h,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
#         for i in range(K):
#             H[i] = np.roll(h_,i+1)
#             if i < (CP-1):
#                 A[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])
#         bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,)) #label
#         # signal_output, sigma2 = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag)
#         signal_output, sigma2,_ = ofdm_simulate_cp_free(bits, H, A, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers)
#         yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
#         #Yp_complex = DFT(yp_complex)
#         Yp_complex = Normalized_FFT_Matrix(K) @ yp_complex
#         yd = signal_output[2*K:4*K].reshape(2*K,1)
#         #H_est,_ = MMSE_CE(Yp_complex,pilotValue,pilotCarriers,K,P,h,SNR)
#         H_est = LS_CE(Yp_complex,pilotValue,pilotCarriers,K,P,interpolate_method)
#         H_true = np.fft.fft(h,n=K)
#         print(H_true)
#         print(H_est)
#         downsampler = allCarriers[::K//CP] #靠谱否？ 补零后FFT的逆运算为抽值后IFFT,应该是靠谱的
#         H_est = H_est[downsampler]
#         h_est = IDFT(H_est) #??64-->16
#         # FH = np.conj(Normalized_FFT_Matrix(CP)).T
#         # h_est = FH @ H_est
#         # print("CIR:",h)
#         # print("h_est:",h_est)
#         H_hat = np.zeros((K,K),dtype=complex)
#         A_hat = np.zeros((K,K),dtype=complex)
#         h_ = np.flip(np.append(h_est,np.zeros((K-CP,1))))  #[0,...,0,hL-1,...h0]
#         for i in range(K):
#             H_hat[i] = np.roll(h_,i+1)
#             if i < (CP-1):
#                 A_hat[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])
#         #以上问题不大（但ofdm_simulate仍值得商榷）
#         FH = np.conj(Normalized_FFT_Matrix(K)).T
#         H_bar = (H_hat-A_hat) @ FH
#         # print(H_hat-A_hat)
#         # print(FH)
#         #print(H_bar)
#         #print(sigma2)
#         #convert complex into real
#         H_bar = np.concatenate(( np.concatenate((np.real(H_bar),-np.imag(H_bar)),axis=1),np.concatenate((np.imag(H_bar),np.real(H_bar)),axis=1) ))
#         x_hat_demod = OAMP(K,yd,H_bar,sigma2,mu=6)
#         #以下问题不大

#         err_bits = np.sum(np.not_equal(x_hat_demod,bits))
#         total_err_bits  += err_bits
#         total_bits += mu*K
#         if err_bits > 0:
#             print("total_err_bits:", total_err_bits,"total_bits:",total_bits,"BER:",total_err_bits/total_bits)
#         if total_err_bits > err_bits_target:
#             end = time.time()
#             print("SNR=",SNR,"iter_time:",end-start)
#             total_time += end-start
#             ber = total_err_bits/total_bits
#             print("BER:", ber)
#             BER.append(ber)
#             break
# print(BER)
# print(total_time)
# BER_matlab = np.array(BER)
# import scipy.io as sio
# sio.savemat('BER.mat', {'BER':BER_matlab})
