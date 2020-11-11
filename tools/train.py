#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf
from .problems import MIMO_detection_problem
from .raputil import Demodulation,Demodulation_16,Demodulation_64

#save 模型/网络中的各变量
def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other

def setup_training(layer_info,prob,lr=1e-3,refinements=(.5,.1,.01),final_refine=None):

    training_stages=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    lr_ = tf.Variable(lr,name='lr',trainable=False)
    #for name,x_hat,var_list,P0,P1,tau_sqr,r in layer_info:
    for name,x_hat,var_list in layer_info:
        loss_ = tf.nn.l2_loss(x_hat - prob.x_)
        #出x_hat或者BER-- 出x_hat，考虑到最后误码率的计算
        if var_list is not None:
            train = tf.train.AdamOptimizer(lr_).minimize(loss_,var_list=var_list)
            #training_stages.append((name,x_hat,loss_,train,var_list,P0,P1,tau_sqr,r))
            training_stages.append((name,x_hat,loss_,train,var_list))
    if final_refine:
        train2_ = tf.train.AdamOptimizer(lr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,x_hat,loss_,train2_,()) )
            #TODO:SGD
    return training_stages

def do_training(training_stages,prob,savefile,sigma2_rayleigh,ivl=1,maxit=1000,better_wait=10,total_batch=5,Mr=4,Nt=4,batch_size=1000):########

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #一开始就有变量用到x_,y_,H_,所以务必要先feed一下
    #sess.run(tf.global_variables_initializer(),feed_dict={prob.y_:prob.ygen_,prob.x_:prob.xgen_,prob.H_:prob.Hgen_})
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess,savefile)

    done=state.get('done',[])
    log=str(state.get('log',''))

#    prob_ = MIMO_detection_problem(sigma2_rayleigh,Mr=Mr,Nt=Nt,SNR=SNR,sample_size=sample_size)
#    yval,xval,Hval,sigma2val = prob_(sess)
    y,x,H,sigma2 = prob(sess)    #prob是TFGenerator的实例，prob(sess)即运行sess.run( ( self.ygen_,self.xgen_ ) )

    #for name,x_hat,loss_,train,var_list,P0,P1,tau_sqr,r in training_stages:
    for name,x_hat,loss_,train,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        loss_history=[]
        for i in range(maxit+1):    #1000 epochs for a layer
            if i%ivl == 0:  #validation:don't use optimizer
                #x_hat_,loss,P0_,P1_,tau_sqr_,r_ = sess.run([x_hat,loss_,P0,P1,tau_sqr,r],feed_dict={prob.y_:y,prob.x_:x,prob.H_:H})    #1000 samples and labels
                loss = sess.run(loss_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval,prob.H_:prob.Hval,prob.sigma2_:prob.sigma2val,prob.sample_size_:prob.sample_sizeval})    #1000 samples and labels
                if np.isnan(loss):
                    raise RuntimeError('loss is NaN')
                loss_history = np.append(loss_history,loss)
                loss_best = loss_history.min()
                sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'.format(i=i,loss=loss,best=loss_best))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(loss_history) - loss_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl >= better_wait:
                        print('move along')
                        break # if it has not improved on the best answer for quite some time, then move along
            for m in range(total_batch): #5 batch, batch_size = 1000 sample
                sess.run(train,feed_dict={prob.y_:y[m*batch_size:(m+1)*batch_size],prob.x_:x[m*batch_size:(m+1)*batch_size],prob.H_:H[m*batch_size:(m+1)*batch_size],prob.sigma2_:sigma2[m*batch_size:(m+1)*batch_size],prob.sample_size_:batch_size})   #1000 samples and labels
                #sess.run(train,feed_dict={prob.y_:prob.ygen_[m*batch_size:(m+1)*batch_size],prob.x_:prob.xgen_[m*batch_size:(m+1)*batch_size],prob.H_:prob.Hgen_[m*batch_size:(m+1)*batch_size],prob.sigma2_:prob.sigma2gen_[m*batch_size:(m+1)*batch_size]})
        done = np.append(done,name)

        log =  log+'\n{name} loss={loss:.9f} in {i} iterations'.format(name=name,loss=loss,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

    return sess

def test(sess,prob,x_hat_T,sigma2_rayleigh,Mr=4,Nt=4,sample_size=1000,mu=2,err_bits_target=1000,SNR=30):

#    BER = []
#    for SNR in range(0,40,5):
    total_err_bits = 0
    total_bits = 0
    while True:
        H,x,y,sigma2 = MIMO_detection_problem(sigma2_rayleigh,mu=mu,Mr=Mr,Nt=Nt,SNR=SNR,validation_size=sample_size,test_flag=True)
        #y,x,H,sigma2 = prob_(sess)
        x_hat_T_ = sess.run(x_hat_T,feed_dict={prob.y_:y,prob.x_:x,prob.H_:H,prob.sigma2_:sigma2,prob.sample_size_:sample_size})
        x = x.reshape(sample_size,2*Nt)
        x_hat_T_ = x_hat_T_.reshape(sample_size,2*Nt)
        x_true = []
        x_pred = []
        for i in range(Nt):
            x_true = np.concatenate((x_true,x[:,i]+1j*x[:,i+Nt]))
            x_pred = np.concatenate((x_pred,x_hat_T_[:,i]+1j*x_hat_T_[:,i+Nt]))
        if mu == 2:
            x_true = Demodulation(x_true)
            x_pred = Demodulation(x_pred)
        elif mu == 4:
            x_true = Demodulation_16(x_true)
            x_pred = Demodulation_16(x_pred)
        else:
            x_true = Demodulation_64(x_true)
            x_pred = Demodulation_64(x_pred)
        err_bits = np.sum(np.not_equal(x_pred,x_true))
        total_err_bits += err_bits
        total_bits += mu*Nt*sample_size
        if err_bits > 0:
            print("total_err_bits:", total_err_bits,"total_bits:",total_bits,"BER:",total_err_bits/total_bits)
        if total_err_bits > err_bits_target:
            print("SNR=",SNR)
            ber = total_err_bits/total_bits
            print("BER:", ber)
            #BER.append(ber)
            break
    # print(BER)
    # BER_matlab = np.array(BER)
    # import scipy.io as sio
    # sio.savemat('BER.mat', {'BER':BER_matlab})

    return ber