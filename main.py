#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to
a) select a problem to be solved
b) select a network type
c) train the network to minimize recovery MSE

"""
import numpy as np
import os

import tensorflow as tf

# np.random.seed(1) # numpy is good about making repeatable output
# tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train,raputil

K = 64
mu = 4
SNR_train = [5,10,15,20,25,30,35,40]
union_test = True
trainOAMP = False
training_epochs = 1000
batch_size = 100
sample_size = 1000
total_batch = 1

BER = []
prob = []
x_hat_T = []
# sess,input_holder,output = networks.build_CE(K,SNR_train[7],savefile='CE-Net_SNR_35dB.npz',test_flag=True)

for i in range(8):
	print("SNR=",SNR_train[i])
	sess,input_holder,output = networks.build_CE(K,SNR_train[i],training_epochs = training_epochs,\
		batch_size=batch_size,savefile='CE-Net_16QAM_SNR_'+str(SNR_train[i])+'dB.npz',test_flag=True)

	# prob = problems.SISO_OFDM_detection_problem(K)

	# input_holder,output = networks.build_CE(K,SNR_train[i],\
	# 	savefile='CE-Net_16QAM_SNR_'+str(SNR_train[i])+'dB.npz',test_flag=True,union_test=union_test,trainOAMP=trainOAMP)

	# if i<4:
	# 	sess,x_hat_T = networks.build_OAMP(prob,T=4,savefile='OAMP2_bg_giid_16QAM64MIMO'+str(i+1)+'.npz',\
	# 		Mr=K,Nt=K,mu=mu,version=1,maxit=1000,better_wait=2000,total_batch=total_batch,batch_size=int(sample_size/total_batch),\
	# 		union_test=union_test,savefileCE='CE-Net_16QAM_SNR_'+str(SNR_train[i])+'dB.npz',\
	# 		trainOAMP=trainOAMP,SNR=SNR_train[i],input_holder=input_holder,output=output)
	# else:
	# 	sess,x_hat_T = networks.build_OAMP(prob,T=4,savefile='OAMP2_bg_giid_16QAM64MIMO.npz',\
	# 		Mr=K,Nt=K,mu=mu,version=1,maxit=1000,better_wait=2000,total_batch=total_batch,batch_size=int(sample_size/total_batch),\
	# 		union_test=union_test,savefileCE='CE-Net_16QAM_SNR_'+str(SNR_train[i])+'dB.npz',\
	# 		trainOAMP=trainOAMP,SNR=SNR_train[i],input_holder=input_holder,output=output)

	ber = raputil.test_DL_OAMP(sess,prob,x_hat_T,input_holder,output,SNR_train[i],OAMPnet=False)
	BER.append(ber)

	tf.reset_default_graph()

print(BER)
BER_matlab = np.array(BER)
import scipy.io as sio
sio.savemat('BER_DL_OAMP_16QAM_match_2_4.mat', {'BER_DL_OAMP_16QAM_match_2_4':BER_matlab})  