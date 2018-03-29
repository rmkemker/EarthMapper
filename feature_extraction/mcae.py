#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 07:36:14 2017

@author: rmk6217
"""

import tensorflow as tf
import numpy as np
import os
import math


class EMA(object):
    def __init__(self, decay=0.99):
        self.avg = None
        self.decay = decay
        self.history = []
    def update(self, X):
        if not self.avg:
            self.avg = X
        else:
            self.avg = self.avg * self.decay + (1-self.decay) * X
        self.history.append(self.avg)
    
def pelu(input_tensor, a , b):
    
    positive = tf.nn.relu(input_tensor) * a / (b + 1e-9)
    negative = a * (tf.exp((-tf.nn.relu(-input_tensor)) / (b + 1e-9)) - 1)
    return (positive + negative)

def bn_pelu(input_tensor, train_tensor, name_sub=''):
    bn = tf.layers.batch_normalization(input_tensor, fused=True, training=train_tensor,
                                       name='bn'+name_sub)
    a = tf.Variable(tf.ones(1), name='a'+name_sub)
    b = tf.Variable(tf.ones(1), name='b'+name_sub)
    return pelu(bn , a, b), a, b

def bn_pelu_conv(input_tensor, train_tensor, scope_id, num_filters,
                 kernel_size=(3,3), strides=(1,1), l2_loss=0.0):
    
    with tf.variable_scope(scope_id):
        act, a, b = bn_pelu(input_tensor, train_tensor)
        conv = tf.layers.conv2d(act, num_filters, kernel_size , padding='same',
                                name='conv2d',
                                kernel_initializer=tf.glorot_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss))
        
    return conv, a, b

def mse_loss(x, y):
    diff = tf.squared_difference(x,y)
    return tf.reduce_mean(diff)

def refinement_layer(lateral_tensor, vertical_tensor, train_tensor, num_filters,
                     scope_id, l2_loss):
    with tf.variable_scope(scope_id):
    
        conv1 = tf.layers.conv2d(lateral_tensor, num_filters, kernel_size=(3,3), 
                                 padding='same',name='conv2d_1',
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss))  
        act1, a1, b1 = bn_pelu(conv1, train_tensor, name_sub='1')
        conv2 = tf.layers.conv2d(act1, num_filters, kernel_size=(3,3), 
                                 padding='same',name='conv2d_2',
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss)) 
        conv3 = tf.layers.conv2d(vertical_tensor, num_filters, kernel_size=(3,3), 
                                 padding='same',name='conv2d_3',
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss))    
        add = tf.add(conv2, conv3)
        
        old_shape = tf.shape(add)[1:3]
        up = tf.image.resize_nearest_neighbor(add, old_shape * 2 )
        act2, a2, b2 = bn_pelu(up, train_tensor, name_sub='2')
        return act2, [a1, a2], [b1, b2]
        

def mcae(input_tensor, train_tensor, nb_bands, cost_weights, l2_loss):
    
    a = []
    b = []
    
    conv1, a_t, b_t = bn_pelu_conv(input_tensor, train_tensor, scope_id='conv1',
                         num_filters=256, l2_loss=l2_loss)
    a.append(a_t)
    b.append(b_t)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
    
    conv2, a_t, b_t = bn_pelu_conv(pool1, train_tensor, scope_id='conv2',
                         num_filters=512, l2_loss=l2_loss)
    a.append(a_t)
    b.append(b_t)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
    
    conv3, a_t, b_t = bn_pelu_conv(pool2, train_tensor, scope_id='conv3',
                         num_filters=512, l2_loss=l2_loss)
    a.append(a_t)
    b.append(b_t)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')
    
    conv4, a_t, b_t = bn_pelu_conv(pool3, train_tensor, scope_id='conv4',
                         num_filters=1024, kernel_size=(1,1), strides=(1,1),
                         l2_loss=l2_loss)
    a.append(a_t)
    b.append(b_t)
    
    refinement3, a_t, b_t = refinement_layer(pool3, conv4, train_tensor, 512,
                                   scope_id='refinement3', l2_loss=l2_loss)
    a+=a_t
    b+=b_t
    
    refinement2, a_t, b_t = refinement_layer(pool2, refinement3, train_tensor, 512,
                                   scope_id='refinement2', l2_loss=l2_loss)
    a+=a_t
    b+=b_t
    
    refinement1, a_t, b_t = refinement_layer(pool1, refinement2, train_tensor, 256,
                                   scope_id='refinement1', l2_loss=l2_loss)
    a+=a_t
    b+=b_t
    
    output_tensor = tf.layers.conv2d(refinement1, nb_bands, kernel_size=(1,1), 
                                 padding='same',name='output_conv',
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss))
    
    loss = cost_weights[0] * mse_loss(output_tensor,input_tensor)
    loss += (cost_weights[1] * mse_loss(conv1,refinement1))
    loss += (cost_weights[2] * mse_loss(conv2,refinement2))   
    loss += (cost_weights[3] * mse_loss(conv3,refinement3))  
    
    loss = loss - 1000.0 * tf.minimum( tf.reduce_min(a), 0)
    loss = loss - 1000.0 * tf.minimum( tf.reduce_min(b), 0) 

    return loss, refinement1


class MCAE(object):
    """Convolutional Autoencoder with single- and multi-loss support.
    
    """
    def __init__(self, nb_bands, loss_weights, ckpt_path = 'checkpoints',
                 num_epochs=1000, starting_learning_rate=2e-3, patience=10, 
                 weight_decay=0.0):
        """
        Args:
            nb_bands (int): Number of bands (dimensionality).
            weight_decay (float): (Default: 1e-4)
            gpu_list (list of int, None): List of GPU ids that the model is trained on.
                If None, the model is trained on all available GPUs.
            model_checkpoint_path (str): Filepath to save the model file after every epoch.
            loss_mode (str): Specifies whether the CAE is single- or multi-loss.
                Supported values are `single` and `multi`.
                (Default: `multi`)
            weights (str): Filepath to the weights file to load. (Default: None)
            loss (str): Loss function used for MCAE.
                Supported functions are `mse` and `mae`.

        """
        self.nb_bands = nb_bands
        self.ckpt_path = ckpt_path
        self.num_epochs = num_epochs
        self.starting_learning_rate = starting_learning_rate
        self.patience = patience
        self.lr_patience = int(patience/2)
        with tf.Graph().as_default():
            inputs = tf.placeholder(tf.float32, shape=(None, None,None,nb_bands),name='inputs')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            training = tf.placeholder(tf.bool,name='training')
            
            tf.add_to_collection("inputs", inputs)
            tf.add_to_collection("training", training)
            tf.add_to_collection('learning_rate', learning_rate)
            
            loss, hidden_output = mcae(inputs, training, nb_bands, loss_weights, weight_decay)
            tf.add_to_collection('loss', loss)
            tf.add_to_collection('hidden_output', hidden_output)
            
            opt = tf.contrib.opt.NadamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = opt.minimize(loss)
            tf.add_to_collection('train_step', train_step)
                
            with tf.Session() as sess:
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                saver.save(sess, self.ckpt_path+'/model.ckpt', 0)        

    def fit(self, X_train, X_val, batch_size=128):
                
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        
        with tf.Graph().as_default():
        
            with tf.Session() as sess:            
                ckpt = tf.train.get_checkpoint_state(self.ckpt_path+'/')  # get latest checkpoint (if any)
                if ckpt and ckpt.model_checkpoint_path:
                    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
                    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta',
                                                       clear_devices=True)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    inputs = tf.get_collection('inputs')[0]
                    training = tf.get_collection('training')[0]
                    train_step = tf.get_collection('train_step')[0]
                    loss = tf.get_collection('loss')[0]
                    learning_rate = tf.get_collection('learning_rate')[0]
                else:
                    raise FileNotFoundError('Cannot Find Checkpoint')
                            
                best_loss = 100000.0            
                lr = self.starting_learning_rate
                
                t_msg1 = '\rEpoch %d (%d/%d) -- lr=%1.0e -- train=%1.2f'
                v_msg1 = ' -- val=%1.2f'    
                pc = 0
                lrc = 0
                
                train_samples = X_train.shape[0]
                idx = np.arange(train_samples)
                train_loss = EMA()
                
                for e in range(self.num_epochs):
                    np.random.shuffle(idx)
                    for i in range(0 , train_samples, batch_size):                    
                        _, loss0 = sess.run([train_step, loss],
                                                feed_dict={inputs: X_train[idx[i:i+batch_size]],
                                                           training: True,
                                                           learning_rate: lr})
                        train_loss.update(loss0)
                        print(t_msg1 % (e+1,i+1,train_samples,lr, train_loss.avg), end="")
                    
                    #Calculate validation loss/acc
                    val_losses = np.array([],dtype=np.float32)
                    for i in range(0, X_val.shape[0], batch_size):
                        val_loss = sess.run([loss],
                                    feed_dict={inputs: X_val[i:i+batch_size],
                                               training: False})
                        val_losses = np.append(val_losses, val_loss)
                    val_loss =  np.mean(val_losses)
    
                    print(v_msg1 % (val_loss), end="")
        
                    if val_loss < best_loss and not math.isnan(val_loss):
                        saver.save(sess, self.ckpt_path+'/model.ckpt',e)
                        best_loss = val_loss
                        print(' -- best_val_loss=%1.2f' % best_loss)
                        pc = 0
                        lrc=0
                    else:
                        pc += 1
                        lrc += 1
                        if pc > self.patience:
                            break
                        elif lrc >= self.lr_patience:
                            lrc = 0
                            lr /= 10.0                        
                        print(' -- Patience=%d/%d' % (pc, self.patience))
                    
                    if math.isnan(val_loss):
                        break    
                
    def transform(self, X, batch_size=128):
        
        n = X.shape[0]
        with tf.Graph().as_default():
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
                  
            with tf.Session() as sess:            
                ckpt = tf.train.get_checkpoint_state(self.ckpt_path+'/')  # get latest checkpoint (if any)
                if ckpt and ckpt.model_checkpoint_path:
                    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
                    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta',
                                                       clear_devices=True)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    inputs = tf.get_collection('inputs')[0]
                    training = tf.get_collection('training')[0]
                    hidden_output = tf.get_collection('hidden_output')[0]
                else:
                    raise FileNotFoundError('Cannot Find Checkpoint')
                            
                if n > batch_size:
                
                    prediction = np.zeros((n,X.shape[1], X.shape[2], 256), dtype=np.float32)
            
                    for i in range(0, n, batch_size):
                        start = np.min([i, n - batch_size])
                        end = np.min([i + batch_size, n])
            
                        prediction[start:end] = sess.run(hidden_output,
                                  feed_dict={inputs: X[start:end],
                                             training: False})
        
                else:
                    prediction = sess.run(hidden_output,
                                  feed_dict={inputs: X,
                                             training: False})
            return prediction
