# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:01:59 2020

@author: 83456

first,  x H y n
second, how to mimic the onsager_deep_learning's framework
  problem: MIMO detection
  network: OAMP(-Net)--compares the structure in both paper with code  and implementation details
  train
"""

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

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

mu = 4
Nt = 64  #4,8,64 QPSK_SNR=15dB 16QAM_SNR=30dB
Mr = Nt
SNR_train = [0,5,10,15,20,25,30,35]


sample_size = 500
total_batch = 50
err_bits_target = 1000
sigma2_rayleigh = 1/Mr
T = 10######
epsilon = 1e-9
channel_type = 0

BER = []

for i in range(1,5):

	print("SNR=",SNR_train[i])
	# Create the basic problem structure.
	if i<6:
	    prob = problems.MIMO_detection_problem(sigma2_rayleigh,mu=mu,Mr=Mr,Nt=Nt,SNR=SNR_train[i],channel_type=channel_type,sample_size=sample_size,validation_size=1000)
	    sess,x_hat_T = networks.build_OAMP(prob,T=4,savefile='OAMP2_bg_giid_16QAM64MIMO'+str(i)+'.npz',lr=1e-3,Mr=Mr,Nt=Nt,mu=mu,version=1,maxit=500,better_wait=1000,total_batch=total_batch,batch_size=int(sample_size/total_batch))
	else:
	    prob = problems.MIMO_detection_problem(sigma2_rayleigh,mu=mu,Mr=Mr,Nt=Nt,SNR=SNR_train[i],channel_type=channel_type,sample_size=sample_size,validation_size=10000)
	    sess,x_hat_T = networks.build_OAMP(prob,T=4,savefile='OAMP2_bg_giid_16QAM64MIMO'+str(i)+'.npz',lr=1e-3,Mr=Mr,Nt=Nt,mu=mu,version=1,maxit=100,better_wait=1000,total_batch=total_batch,batch_size=int(sample_size/total_batch))

	# build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
	# if i < 4:
	# 	layers,x_hat_T = networks.build_OAMP(prob,T=4,Mr=Mr,Nt=Nt,mu=mu,version=0)
	# else:
	# 	layers,x_hat_T = networks.build_OAMP(prob,T=10,Mr=Mr,Nt=Nt,mu=mu,version=0)
	

	# plan the learning########
	# if i<6:
	# 	training_stages = train.setup_training(layers,prob,lr=1e-3)
	# else:
	# 	training_stages = train.setup_training(layers,prob,lr=1e-4)
	# training_stages = train.setup_training(layers,prob,lr=1e-3)

	# # do the learning (takes a while)
	# sess = train.do_training(training_stages,prob,'OAMP1_bg_giid'+str(i)+'.npz',sigma2_rayleigh,Mr=Mr,Nt=Nt,total_batch=total_batch,batch_size=int(sample_size/total_batch))

	#TODO: test the model
	ber = train.test(sess,prob,x_hat_T,sigma2_rayleigh,Mr=Mr,Nt=Nt,mu=mu,sample_size=int(sample_size/total_batch),err_bits_target=err_bits_target,SNR=SNR_train[i])
	BER.append(ber)

	tf.reset_default_graph()

print(BER)
BER_matlab = np.array(BER)
import scipy.io as sio
sio.savemat('BER_16QAM64MIMO_2.mat', {'BER_16QAM64MIMO_2':BER_matlab})
