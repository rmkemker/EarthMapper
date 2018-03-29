#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:19:42 2017

@author: Ronald Kemker
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import convolve

def meanPooling(X, poolSize):
    """
    meanPooling - Average pooling filter across data
    Input:  data - [height x width x channels]
            poolSize - (int) Receptive field size of mean pooling filter
    Output: output - [height x width x channels]
    """
    X = np.float32(X[np.newaxis])
    with tf.Graph().as_default():
        X_t = tf.placeholder(tf.float32, shape=X.shape)
        pool_filter = tf.ones((poolSize, poolSize, X.shape[-1], 1), tf.float32)/poolSize**2
        half = int(poolSize/2)
        pad_t = tf.pad(X_t , tf.constant([[0,0],[half, half], [half, half],[0,0]]), 'REFLECT')
        pool_op = tf.nn.depthwise_conv2d(pad_t, pool_filter, strides=[1,1,1,1], padding='VALID')
        with tf.Session() as sess:
            return sess.run(pool_op, {X_t:X})[0]
    
def conv2d(X, w):
    
    with tf.Graph().as_default():
        X = np.float32(X[np.newaxis])
        X_t = tf.placeholder(tf.float32, name='X_t') 
        w_t = tf.placeholder(tf.float32, name='w_t')
        
        conv_op = tf.nn.conv2d(X_t, w_t, strides=[1,1,1,1], padding='SAME')
        
        nb_filters = w.shape[3]
        
        with tf.Session() as sess:
            
            if nb_filters <= 64:
                return sess.run(conv_op, {X_t:X, w_t:w})[0]
            else:
                output = np.zeros((X.shape[1], X.shape[2], nb_filters))
                for i in range(0, nb_filters, 64):
                    idx = np.arange(i,np.min([i+64, nb_filters]))
                    output[:,:,idx] = sess.run(conv_op, {X_t:X,w_t:w[:,:,:,idx]})[0]
                return output[0]
            
def mean_pooling(X, pool_size):
    
    X = np.float32(X)
    sh = X.shape
    f = np.ones((pool_size, pool_size) , dtype=np.float32) / pool_size**2
       
    result = np.zeros(sh, dtype=np.float32)
    
    for i in range(sh[-1]):
        result[:,:,i] = convolve(X[:,:,i], f, mode='reflect')
    return result

if __name__ == "__main__":
    
    X = np.float32(np.random.rand(610,340,256*5*3))
    result = mean_pooling(X, 5)
    