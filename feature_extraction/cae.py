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


class DataSet(object):

    def __init__(self, images):
                
        self._num_examples = images.shape[0]
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._index_array = np.arange(self._num_examples)
        np.random.shuffle(self._index_array)

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples


    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            np.random.shuffle(self._index_array)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if batch_size <= self._num_examples:
            return self._images[self._index_array[start:end]]
        else:
            return self._images
    

def bn_relu(input_tensor, train_tensor, name_sub=''):
    bn = tf.layers.batch_normalization(input_tensor, fused=True, training=train_tensor,
                                       name='bn'+name_sub)
    return tf.nn.relu(bn, name='act'+name_sub)

def bn_relu_conv(input_tensor, train_tensor, scope_id, num_filters,
                 kernel_size=(3,3), strides=(1,1), l2_loss=0.0):
    
    with tf.variable_scope(scope_id):
        act= bn_relu(input_tensor, train_tensor)
        conv = tf.layers.conv2d(act, num_filters, kernel_size , padding='same',
                                name='conv2d',
                                kernel_initializer=tf.glorot_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss))
        
    return conv

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
        act1 = bn_relu(conv1, train_tensor, name_sub='1')
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
        act2 = bn_relu(up, train_tensor, name_sub='2')
        return act2
        

def cae(N , input_tensor, train_tensor, nb_bands, cost_weights, l2_loss):
    

    conv1 = bn_relu_conv(input_tensor, train_tensor, scope_id='conv1',
                         num_filters=N, l2_loss=l2_loss)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
    
    conv2 = bn_relu_conv(pool1, train_tensor, scope_id='conv2',
                         num_filters=N*2, l2_loss=l2_loss)
    
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
    
    conv3 = bn_relu_conv(pool2, train_tensor, scope_id='conv3',
                         num_filters=N*2, l2_loss=l2_loss)

    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')
    
    conv4 = bn_relu_conv(pool3, train_tensor, scope_id='conv4',
                         num_filters=N*4, kernel_size=(1,1), strides=(1,1),
                         l2_loss=l2_loss)

    refinement3 = refinement_layer(pool3, conv4, train_tensor, N*2,
                                   scope_id='refinement3', l2_loss=l2_loss)

    
    refinement2 = refinement_layer(pool2, refinement3, train_tensor, N*2,
                                   scope_id='refinement2', l2_loss=l2_loss)
    
    refinement1 = refinement_layer(pool1, refinement2, train_tensor, N,
                                   scope_id='refinement1', l2_loss=l2_loss)
    
    output_tensor = tf.layers.conv2d(refinement1, nb_bands, kernel_size=(1,1), 
                                 padding='same',name='output_conv',
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss))
    
    loss =  mse_loss(output_tensor,input_tensor)

    return loss, refinement1


class CAE(object):
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
        self.N = 64
        self.nb_bands = nb_bands
        self.ckpt_path = ckpt_path
        self.num_epochs = num_epochs
        self.starting_learning_rate = starting_learning_rate
        self.patience = patience
        self.lr_patience = int(patience/2)
        
        inputs = tf.placeholder(tf.float32, shape=(None, None,None, nb_bands),name='inputs')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        training = tf.placeholder(tf.bool,name='training')
        
        tf.add_to_collection("inputs", inputs)
        tf.add_to_collection("training", training)
        tf.add_to_collection('learning_rate', learning_rate)
        
        loss, hidden_output = cae(self.N , inputs, training, nb_bands, loss_weights, weight_decay)
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
        
        train_set = DataSet(X_train)
        val_set = DataSet(X_val)
        
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
                train_step = tf.get_collection('train_step')[0]
                loss = tf.get_collection('loss')[0]
                learning_rate = tf.get_collection('learning_rate')[0]
            else:
                raise FileNotFoundError('Cannot Find Checkpoint')
            
            iter_per_epoch = int(X_train.shape[0] / batch_size)
            val_iter = int(val_set.num_examples / batch_size) + 1    
            
            best_loss = 100000.0            
            loss_arr = np.zeros((iter_per_epoch),np.float32)
            lr = self.starting_learning_rate
            
            t_msg1 = '\rEpoch %d/%d (%d/%d) -- lr=%1.0e'
            t_msg2 = ' -- train=%1.2f'
            v_msg1 = ' -- val=%1.2f'    
            pc = 0
            lrc = 0
            for e in range(self.num_epochs):
                for i in range(iter_per_epoch):
                    print(t_msg1 % (e+1,self.num_epochs,i+1,iter_per_epoch,lr), end="")
                    images = train_set.next_batch(batch_size)
                    _, loss0 = sess.run([train_step, loss],
                                            feed_dict={inputs: images,
                                                       training: True,
                                                       learning_rate: lr})
                    loss_arr[i] = loss0
                print(t_msg2 % (np.mean(loss_arr)), end="")
                
                #Calculate validation loss/acc
                val_losses = np.array([],dtype=np.float32)
                for i in range(val_iter):
                    X = val_set.next_batch(batch_size)
                    val_loss = sess.run([loss],
                                feed_dict={inputs: X,
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
        
        dataset = DataSet(X)
        n = dataset._num_examples
        
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
            
                prediction = np.zeros((dataset._num_examples,X.shape[1], X.shape[2], self.N), dtype=np.float32)
        
                for i in range(0, n, batch_size):
                    start = np.min([i, n - batch_size])
                    end = np.min([i + batch_size, n])
        
                    X_test = dataset.images[start:end]
        
                    prediction[start:end] = sess.run(hidden_output,
                              feed_dict={inputs: X_test,
                                         training: False})
    
            else:
                prediction = sess.run(hidden_output,
                              feed_dict={inputs: X,
                                         training: False})
            return prediction
