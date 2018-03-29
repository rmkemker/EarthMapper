#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:58:44 2017

@author: rmk6217
"""

import numpy as np

def random_data_split(y, train_samples, classwise=True, ignore_zero=True,
                      shuffle=False):
    """Split label data into two folds.

    Args:
        y (array of ints): M x N array of labels, from [0, N]
        train_samples (float, int): Determines how many samples will go to the
            training fold. Treated as percentage if float, treated as absolute
            if int.
        classwise (boolean): if True, training samples will be picked with
            respect to the classes rather than with respect to the image.
        ignore_zero (boolean): if True, neither fold will contain the 0 class.

    Returns:
        Array of ints, both M x N. Samples allocated for the other fold are
        set to -1. If `ignore_zero`, all values are decremented by 1.
    """


    if not isinstance(train_samples, np.ndarray):
        relative = isinstance(train_samples, float) and 0 <= train_samples <= 1
        if not relative:
            cutoff = train_samples
        
    y = np.array(y)
    sh = y.shape
    y = y.ravel()
    
    num_classes = np.max(y) + 1
    
    y_train = np.ones(np.prod(sh), dtype=np.int32) * -1
    y_test = np.ones(np.prod(sh), dtype=np.int32) * -1

    if isinstance(train_samples, np.ndarray):
        start = 1 if ignore_zero else 0
        for i in range(start, num_classes):        
            idx = np.where(y == i)[0]
            if shuffle:
                np.random.shuffle(idx)
            y_train[idx[:train_samples[i]]] = i
            y_test[idx[train_samples[i]:]] = i
    
    elif classwise:
        start = 1 if ignore_zero else 0
        for i in range(start, num_classes):
            idx = np.where(y == i)[0]
            if relative:
                cutoff = int(round(len(idx) * train_samples))
            if shuffle:
                np.random.shuffle(idx)
            y_train[idx[:cutoff]] = i
            y_test[idx[cutoff:]] = i
    else:
        if ignore_zero:
            idx = np.where(y > 0)[0]
        else:
            idx = np.arange(len(y))
        if shuffle:
            np.random.shuffle(idx)
        if relative:
            cutoff = int(round(len(idx) * train_samples))
        y_train[idx[:cutoff]] = y[idx[:cutoff]]
        y_test[idx[cutoff:]] = y[idx[cutoff:]]
    y_train = y_train.reshape(sh)
    y_test = y_test.reshape(sh)
    
    if ignore_zero:
        y_train[y_train > 0] -= 1
        y_test[y_test > 0] -= 1
    
    return y_train, y_test


def train_val_split(y, train_samples, val_samples, classwise=True, ignore_zero=True):
    """Randomly split label data into two folds.

    Args:
        y (array of ints): M x N array of labels, from [0, N]
        train_samples (int): Determines how many samples will go to the
            training fold. 
        val_samples (int): Determines how many samples will go to the
            validation fold.             
        classwise (boolean): if True, training samples will be picked with
            respect to the classes rather than with respect to the image.
        ignore_zero (boolean): if True, neither fold will contain the 0 class.

    Returns:
        Array of ints, both M x N. Samples allocated for the other fold are
        set to -1. If `ignore_zero`, all values are decremented by 1.
    """

       
    y = np.array(y)
    sh = y.shape
    y = y.ravel()
    
    num_classes = np.max(y) + 1
    
    y_train = np.ones(np.prod(sh), dtype=np.int32) * -1
    y_val = np.ones(np.prod(sh), dtype=np.int32) * -1

    if isinstance(train_samples, np.ndarray):

        train_samples = np.int32(train_samples)
        val_samples = np.int32(val_samples)
        start = 1 if ignore_zero else 0
        for i in range(start, num_classes):        
            idx = np.where(y == i)[0]
            np.random.shuffle(idx)
            y_train[idx[:train_samples[i]]] = i
            y_val[idx[train_samples[i]:train_samples[i]+val_samples[i]]] = i
    
    elif classwise:
        start = 1 if ignore_zero else 0
        train_samples = np.int32(train_samples)
        val_samples = np.int32(val_samples)
        for i in range(start, num_classes):
            idx = np.where(y == i)[0]

            np.random.shuffle(idx)
            y_train[idx[:train_samples]] = i
            y_val[idx[train_samples:train_samples+val_samples]] = i
    else:
        if ignore_zero:
            idx = np.where(y > 0)[0]
        else:
            idx = np.arange(len(y))
        np.random.shuffle(idx)

        y_train[idx[:train_samples]] = y[idx[:train_samples]]
        y_val[idx[train_samples:train_samples+val_samples]] = y[idx[train_samples:train_samples+val_samples]]
    y_train = y_train.reshape(sh)
    y_val = y_val.reshape(sh)
    
    if ignore_zero:
        y_train[y_train > 0] -= 1
        y_val[y_val > 0] -= 1
    
    return y_train, y_val

