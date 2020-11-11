#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf
import tools.shrinkage as shrinkage
from .train import load_trainable_vars,save_trainable_vars
from .raputil import get_WMMSE,sample_gen,sample_gen_for_OAMP

def build_CE(K,SNR,savefile,learning_rate=1e-3,training_epochs=2000,batch_size=50,\
    test_flag=False,union_test=False,trainOAMP=False):    
    #input Hls:1*2K ; output Hout:1*2K
    n_input = 2*K
    n_output = 2*K

    H_LS = tf.placeholder(tf.float64,(None,n_input),name='H_LS')    #input 
    H_true = tf.placeholder(tf.float64,(None,n_output),name='H_true')   #label

    #get initial weight
    W_MMSE_init = get_WMMSE(SNR)
    if trainOAMP:
        W_MMSE = tf.Variable(W_MMSE_init,dtype=tf.float64,name='W_MMSE',trainable=False)    #WMMSE_init 2K*2K
        bias = tf.Variable(np.zeros(n_input),dtype=tf.float64,name='bias',trainable=False)  #可调参数量非常大，适合随时记录吗？
    else:
        W_MMSE = tf.Variable(W_MMSE_init,dtype=tf.float64,name='W_MMSE')    #WMMSE_init 2K*2K
        bias = tf.Variable(np.zeros(n_input),dtype=tf.float64,name='bias')  #可调参数量非常大，适合随时记录吗？
    H_out = tf.add(tf.matmul(H_LS, W_MMSE), bias)

    if union_test or trainOAMP:
        return H_LS,H_out
    # Define loss and optimizer, minimize the l2 loss
    loss_ = tf.nn.l2_loss(H_out - H_true) #is it right?后面选取模型的指标也是，必须合理
    #cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_,var_list=tf.trainable_variables())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess,savefile)
    log=str(state.get('log',''))
    print(log)

    if test_flag:
        return sess,H_LS,H_out

    test_step = 5
    loss_history=[]
    save={}      #for the best model

    for epoch in range(training_epochs+1):  #does it have mini-batch?

        batch_samples,batch_labels = sample_gen(batch_size,SNR,training_flag=True)
        _,loss = sess.run([optimizer,loss_],feed_dict={H_LS:batch_samples, H_true:batch_labels})
        sys.stdout.write('\repoch={epoch:<6d} loss={loss:.9f} on train set'.format(epoch=epoch,loss=loss))
        sys.stdout.flush()

        if epoch%test_step == 0:
            batch_samples,batch_labels = sample_gen(batch_size,SNR,training_flag=False)
            loss = sess.run(loss_,feed_dict={H_LS:batch_samples, H_true:batch_labels})
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history,loss)
            loss_best = loss_history.min()
            #for the best model
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v) 
            #
            print("\nepoch={epoch:<6d} loss={loss:.9f} (best={best:.9f}) on test set".format(epoch=epoch,loss=loss,best=loss_best))

    tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
    for k,d in save.items():
        if k in tv:
            sess.run(tf.assign( tv[k], d) )
            print('restoring ' + k+' = '+str(d))

    log =  log+'\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} iterations'.format(loss=loss,i=epoch,best=loss_best,j=loss_history.argmin()*test_step)

    state['log'] = log
    save_trainable_vars(sess,savefile,**state)

    print("optimization finished")

    return sess,H_LS,H_out


#参数都不在此文件中定义，需要用的传过来
def build_OAMP(prob,T,savefile,Mr=4,Nt=4,mu=2,version=0,lr=1e-3,\
    maxit=1000,better_wait=100,total_batch=50,batch_size=100,\
    union_test=False,savefileCE='',trainOAMP=False,SNR=20,input_holder=None,output=None):

    layers = []   #layerinfo:(name,xhat_,newvars)

    H_ = prob.H_
    x_ = prob.x_
    y_ = prob.y_
    sigma2 = prob.sigma2_
    sample_size = prob.sample_size_

    # precompute some tensorflow constants
    OneOver2N = tf.constant(float(1)/(2*Nt),dtype=tf.float32)
    NormalConstant = tf.constant(float(1)/(2*np.pi)**0.5,dtype=tf.float32)
    epsilon = tf.constant(1e-2,dtype=tf.float32)
    HT_ = tf.transpose(H_,perm=[0,2,1])
    HHT = tf.matmul(H_,HT_)
    OneOver_trHTH = tf.reshape(1/tf.trace(tf.matmul(HT_,H_)),[sample_size,1,1])
    sigma2Over4N = sigma2/(4*Nt)
    sigma2_I = sigma2/2*tf.eye(2*Mr,batch_shape=[sample_size],dtype=tf.float32)
    I = tf.eye(2*Nt,batch_shape=[sample_size],dtype=tf.float32)

    x_hat = tf.zeros_like(x_,dtype=tf.float32)

    for t in range(T):
        theta_ = tf.Variable(float(1),dtype=tf.float32,name='theta_'+str(t))
        gamma_ = tf.Variable(float(1),dtype=tf.float32,name='gamma_'+str(t))
        if version == 1:
            phi_ = tf.Variable(float(1),dtype=tf.float32,name='phi_'+str(t))
            xi_ = tf.Variable(float(0),dtype=tf.float32,name='xi_'+str(t))
        p_noise = y_-tf.matmul(H_, x_hat)
        v_sqr = (tf.reshape(tf.square(tf.norm(p_noise,axis=(1,2))),[sample_size,1,1]) - Mr*sigma2) * OneOver_trHTH
        v_sqr = tf.maximum(v_sqr,epsilon)
        #with tf.device("/cpu:0"):
        w_hat = v_sqr * tf.matmul(HT_, tf.linalg.inv(v_sqr*HHT+ sigma2_I))
        w = 2*Nt / tf.reshape(tf.trace(tf.matmul(w_hat,H_)),[sample_size,1,1]) * w_hat

        r = x_hat + gamma_*tf.matmul(w,p_noise)
        C = I - theta_*tf.matmul(w,H_)
        tau_sqr = OneOver2N * tf.reshape(tf.trace(tf.matmul(C,tf.transpose(C,perm=[0,2,1]))),[sample_size,1,1]) * v_sqr + tf.square(theta_)*sigma2Over4N * tf.reshape(tf.trace(tf.matmul(w,tf.transpose(w,perm=[0,2,1]))),[sample_size,1,1])
        tau_sqr = tf.maximum(tau_sqr,epsilon)
        if mu == 2: #{-1,+1}
            #clipping
            r = tf.maximum(r,-2.*tf.ones_like(r,dtype=tf.float32))
            r = tf.minimum(r,2.*tf.ones_like(r,dtype=tf.float32))
            P0 = tf.exp(-tf.square(-1-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P1 = tf.exp(-tf.square(1-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            x_hat = (P1-P0) / (P1+P0)
            if version == 1:
                x_hat = phi_ * (x_hat- xi_*r)   #(18)
        elif mu == 4: #{-3,-1,+1,+3}
            #clipping
            r = tf.maximum(r,-4.*tf.ones_like(r,dtype=tf.float32))
            r = tf.minimum(r,4.*tf.ones_like(r,dtype=tf.float32))
            # P_3 = tf.minimum(\
            #     tf.maximum(tf.exp(-tf.square(-3-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr),\
            #     -3.e+38*tf.ones_like(r,dtype=tf.float32)),\
            #     3.e+38*tf.ones_like(r,dtype=tf.float32))
            P_3 = tf.exp(-tf.square(-3-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P_1 = tf.exp(-tf.square(-1-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P1 = tf.exp(-tf.square(1-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P3 = tf.exp(-tf.square(3-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            x_hat = (-3*P_3-P_1+P1+3*P3) / (P_3+P_1+P1+P3)
            if version == 1:
                x_hat = phi_ * (x_hat- xi_*r)   #(18)
        else: #{-1,+1}
            r = tf.maximum(r,-8.*tf.ones_like(r,dtype=tf.float32))
            r = tf.minimum(r,8.*tf.ones_like(r,dtype=tf.float32))
            P_7 = tf.exp(-tf.square(-7-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P_5 = tf.exp(-tf.square(-5-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P_3 = tf.exp(-tf.square(-3-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P_1 = tf.exp(-tf.square(-1-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P1 = tf.exp(-tf.square(1-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P3 = tf.exp(-tf.square(3-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P5 = tf.exp(-tf.square(5-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            P7 = tf.exp(-tf.square(7-r)/(2*tau_sqr)) * NormalConstant / tf.sqrt(tau_sqr)
            x_hat =  (-7*P_7-5*P_5-3*P_3-P_1+P1+3*P3+5*P5+7*P7) / (P_7+P_5+P_3+P_1+P1+P3+P5+P7)
            if version == 1:
                x_hat = phi_ * (x_hat- xi_*r)   #(18)

        #layers.append(('OAMP T={0}'.format(t),x_hat,(theta_,gamma_,),P0,P1,tau_sqr,r))
        if version == 0:
            layers.append(('OAMP T={0}'.format(t),x_hat,(theta_,gamma_,)))
        else:
            layers.append(('OAMP T={0}'.format(t),x_hat,(theta_,gamma_,phi_,xi_,)))

    loss_ = tf.nn.l2_loss(x_hat - x_)
    lr_ = tf.Variable(lr,name='lr',trainable=False)
    if tf.trainable_variables() is not None:
        train = tf.train.AdamOptimizer(lr_).minimize(loss_,var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess,savefile)
    done=state.get('done',[])
    log=str(state.get('log',''))

    for name,_,var_list in layers:
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])
        done = np.append(done,name)
        print(name + ' ' + describe_var_list)
    print(log)

    if union_test:
        state = load_trainable_vars(sess,savefileCE)
        log=str(state.get('log',''))
        print(log) 
        return sess,x_hat 
    if trainOAMP:
        #load
        other={} 
        try: 
            tv=dict([ (str(v.name),v) for v in tf.global_variables() ])
            for k,d in np.load(savefileCE).items():
                if k in tv:
                    print('restoring ' + k)
                    sess.run(tf.assign( tv[k], d) )
                else:
                    other[k] = d
                    #print('err')
            log=str(other.get('log',''))
            print(log)
        except IOError:
            pass   

    loss_history=[]
    save={}      #for the best model
    ivl = 1
    #y,x,H,sigma2 = prob(sess)    #prob是TFGenerator的实例，prob(sess)即运行sess.run( ( self.ygen_,self.xgen_ ) )
    yval,xval,Hval,sigma2val = sample_gen_for_OAMP(batch_size*total_batch,SNR,\
                sess, input_holder,output, training_flag=False)
    y,x,H,sigma2 = sample_gen_for_OAMP(batch_size*total_batch,SNR, sess, input_holder,output)
    for i in range(maxit+1):
        # if i%1000 == 0:
        #     yval,xval,Hval,sigma2val = sample_gen_for_OAMP(batch_size*total_batch,SNR,\
        #         sess, input_holder,output, training_flag=False)
        #     y,x,H,sigma2 = sample_gen_for_OAMP(batch_size*total_batch,SNR, sess, input_holder,output)       
        if i%ivl == 0:  #validation:don't use optimizer            
            loss = sess.run(loss_,feed_dict={prob.y_:yval,
                    prob.x_:xval,prob.H_:Hval,prob.sigma2_:sigma2val,
                    prob.sample_size_:batch_size*total_batch})    #1000 samples and labels
            # loss = sess.run(loss_,feed_dict={prob.y_:prob.yval,
            #         prob.x_:prob.xval,prob.H_:prob.Hval,prob.sigma2_:prob.sigma2val,
            #         prob.sample_size_:prob.sample_sizeval})    #1000 samples and labels
            if np.isnan(loss):
                #print(np.amin(P_3_))
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history,loss)
            loss_best = loss_history.min()
            #for the best model
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v) 
            #
            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'.format(i=i,loss=loss,best=loss_best))
            sys.stdout.flush()
            if i%(100*ivl) == 0:
                print('')
                age_of_best = len(loss_history) - loss_history.argmin()-1 # how long ago was the best nmse?
                if age_of_best*ivl >= better_wait:
                    print('move along')
                    break # if it has not improved on the best answer for quite some time, then move along
        for m in range(total_batch): #5 batch, batch_size = 1000 sample
            # sess.run(train,feed_dict={prob.y_:y,prob.x_:x,prob.H_:H,
            #                             prob.sigma2_:sigma2,prob.sample_size_:batch_size})   #1000 samples and labels
            sess.run(train,feed_dict={prob.y_:y[m*batch_size:(m+1)*batch_size],
                                        prob.x_:x[m*batch_size:(m+1)*batch_size],
                                        prob.H_:H[m*batch_size:(m+1)*batch_size],
                                        prob.sigma2_:sigma2[m*batch_size:(m+1)*batch_size],
                                        prob.sample_size_:batch_size})   #1000 samples and labels
    #done = np.append(done,name)
    #for the best model----it's for the strange phenomenon
    tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
    for k,d in save.items():
        if k in tv:
            sess.run(tf.assign( tv[k], d) )
            print('restoring ' + k+' = '+str(d))
    #
    #log =  log+'\nloss={loss:.9f} in {i} iterations'.format(loss=loss,i=i)
    log =  log+'\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} iterations'.format(loss=loss,i=i,best=loss_best,j=loss_history.argmin())

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess,savefile,**state)

    return sess,x_hat

    #return layers,x_hat
