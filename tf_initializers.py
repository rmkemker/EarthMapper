#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:32:06 2017

@author: Ronald Kemker
"""

import tensorflow as tf
import math
import numpy as np

def get(initializer):
    return eval(initializer) if isinstance(initializer, str) else initializer

def zeros(shape, name=None):
    return tf.Variable(tf.zeros([shape]), name=name)

def ones(shape, name=None):
    return tf.Variable(tf.ones([shape]), name=name)

def constant(shape, value=0, name=None):
    return tf.Variable(value * tf.ones([shape]), name=name)

def glorot_normal(shape, name=None):
    return _normal_base(shape, name, stddev=math.sqrt(2 / np.sum(shape)))

def he_normal(shape, name=None):
    return _normal_base(shape, name, stddev=math.sqrt(2 / shape[0]))

def lecun_normal(shape, name=None):
    return _normal_base(shape, name, stddev=math.sqrt(1 / shape[0]))

def _normal_base(shape, name, stddev):
    return tf.Variable(tf.random_normal(shape, name=name)) * stddev
