#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow as tf
from .raputil import Modulation,Modulation_16,Modulation_64

class Generator_(object):
    def __init__(self,A,**kwargs):
        self.A = A
#        self.sample_size = sample_size
        M,N = A.shape
        vars(self).update(kwargs)   #更新关键字参数
        self.x_ = tf.placeholder( tf.float32,(None,N,1),name='x' )
        self.y_ = tf.placeholder( tf.float32,(None,M,1),name='y' )
        self.H_ = tf.placeholder( tf.float32,(None,M,N),name='H' )
        self.sigma2_ = tf.placeholder( tf.float32,(None,1,1),name='sigma2' )
        self.sample_size_ = tf.placeholder( tf.int32,name='sample_size' )

class TFGenerator_(Generator_):
    def __init__(self,**kwargs):
        Generator_.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_,self.Hgen_,self.sigma2gen_ ) )

def SISO_OFDM_detection_problem(K):
    prob = TFGenerator_(A=np.zeros((2*K,2*K)))
    prob.name = 'SISO_OFDM detection'
    return prob

def MIMO_detection_problem(sigma2_rayleigh,mu=2,Mr=4,Nt=4,SNR=30,channel_type=0,sample_size=1000,validation_size=1000,SNR_flag=False,test_flag=False):  #use the same SNR for train sets & validation sets, is there a problem

    if test_flag == True:
        sample_size = 0
    H_ = np.zeros((2*sample_size*Mr,2*Nt))
    x_ = np.zeros((2*sample_size*Nt,1))
    y_ = np.zeros((2*sample_size*Mr,1))
    sigma2_ = np.zeros((sample_size,1))
    Hval_ = np.zeros((2*validation_size*Mr,2*Nt))
    xval_ = np.zeros((2*validation_size*Nt,1))
    yval_ = np.zeros((2*validation_size*Mr,1))
    sigma2val_ = np.zeros((validation_size,1))
    SNR_ = [0,5,10,15,20,25,30,35]
    for i in range(sample_size+validation_size):

        if channel_type == 0:     #Rayleigh MIMO channel
            H = np.sqrt(sigma2_rayleigh/2)*(np.random.randn(Mr,Nt) + 1j * np.random.randn(Mr,Nt))
    #   else:                     #Correlated MIMO channel

        bits = np.random.binomial(n=1, p=0.5, size=(mu*Nt, ))
#        bits_mod = Modulation(bits)
        if mu == 2:
            bits_mod = Modulation(bits)
        elif mu == 4:
            bits_mod = Modulation_16(bits)
        else:
            bits_mod = Modulation_64(bits)
        x = bits_mod.reshape((Nt,1))
        y = H.dot(x)
        #add AWGN noise
        if SNR_flag == True:
            SNR = SNR_[i%8]
        signal_power = np.mean(abs(y**2))
        sigma2 = signal_power * 10**(-SNR/10)
        noise = np.sqrt(sigma2/2) * (np.random.randn(Mr,1)+1j*np.random.randn(Mr,1))
        y = y + noise
        #convert complex into real
        x = np.concatenate((np.real(x),np.imag(x)))
        H = np.concatenate(( np.concatenate((np.real(H),-np.imag(H)),axis=1),np.concatenate((np.imag(H),np.real(H)),axis=1) ))
        y = np.concatenate((np.real(y),np.imag(y)))
        #stack
        if i<sample_size:
            H_[2*Mr*i:2*Mr*(i+1)] = H
            x_[2*Nt*i:2*Nt*(i+1)] = x
            y_[2*Mr*i:2*Mr*(i+1)] = y
            sigma2_[i] = sigma2
        else:
            Hval_[2*Mr*(i-sample_size):2*Mr*(i-sample_size+1)] = H
            xval_[2*Nt*(i-sample_size):2*Nt*(i-sample_size+1)] = x
            yval_[2*Mr*(i-sample_size):2*Mr*(i-sample_size+1)] = y
            sigma2val_[i-sample_size] = sigma2
    #reshape
    H_ = H_.reshape(sample_size,2*Mr,2*Nt)
    x_ = x_.reshape(sample_size,2*Nt,1)
    y_ = y_.reshape(sample_size,2*Mr,1)
    sigma2_ = sigma2_.reshape(sample_size,1,1)
    Hval_ = Hval_.reshape(validation_size,2*Mr,2*Nt)
    xval_ = xval_.reshape(validation_size,2*Nt,1)
    yval_ = yval_.reshape(validation_size,2*Mr,1)
    sigma2val_ = sigma2val_.reshape(validation_size,1,1)

#    print("Finish Building up Train Sets and Validation Sets")
    if test_flag == False:
        prob = TFGenerator_(A=H)
        prob.name = 'MIMO detection'
        #generate the whole train sets and validation sets at one time(with different channel--time varying)
        prob.xval = xval_
        prob.yval = yval_
        prob.Hval = Hval_
        prob.sigma2val = sigma2val_
        prob.sample_sizeval = validation_size
        #暂存
        prob.xgen_ = tf.convert_to_tensor(x_.astype(np.float64))
        prob.ygen_ = tf.convert_to_tensor(y_.astype(np.float64))
        prob.Hgen_ = tf.convert_to_tensor(H_.astype(np.float64))
        prob.sigma2gen_ = tf.convert_to_tensor(sigma2_.astype(np.float64))
        #prob.sample_sizegen_ = tf.convert_to_tensor(sample_size)

        return prob

    return Hval_,xval_,yval_,sigma2val_
