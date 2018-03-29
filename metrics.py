#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:42:10 2017

@author: rmkemker
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import time 

class Metrics():
    def __init__(self, truth, prediction):        
        self.c = confusion_matrix(truth , prediction)
            
    def _oa(self):
        return np.sum(np.diag(self.c))/np.sum(self.c)
        
    def _aa(self):
        return np.mean(np.diag(self.c)/np.sum(self.c, axis=1))
    
    def _ca(self):
        return np.diag(self.c)/np.sum(self.c, axis=1)

    def per_class_accuracies(self):
        return self._ca()
        
    def mean_class_accuracy(self):
        return self._aa()
    
    def overall_accuracy(self):
        return self._oa()
    
    def _kappa(self):
        e = np.sum(np.sum(self.c,axis=1)*np.sum(self.c,axis=0))/np.sum(self.c)**2             
        return (self._oa()-e)/(1-e)
        
    def standard_metrics(self):
        return self._oa(), self._aa(), self._kappa()
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.tic = time.time() 
    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %1.3f seconds' % (time.time() - self.tic))