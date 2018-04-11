#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:23:03 2018

@author: Ronald Kemker and Utsav Gewali
"""

from fileIO.utils import read
import numpy as np
from metrics import Metrics, Timer
import os
from utils.utils import readENVIHeader
from pipeline import Pipeline
from preprocessing import ImagePreprocessor, AveragePooling2D
from utils.train_val_split import random_data_split, train_val_split
from feature_extraction.scae import SCAE
from classifier.svm_cv_workflow import SVMWorkflow

np.random.seed(6)
gpu_id = "0"
dataset = 'paviau' # dataset choices for this example are 'ip' and 'paviau'
fe_type = 'scae' #feature extractor options are 'scae' and 'smcae'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
root = '/home/rmk6217/Documents/EarthMapper/'
fe_path = root + 'feature_extraction/pretrained/' + fe_type
data_path = root + 'datasets/indian_pines/'
num_trials = 30

#Load Indian Pines or Pavia University dataset
if dataset == 'paviau':
    data_path = root + 'datasets/pavia_university/'
    X = np.float32(read(data_path+'pavia_ds/pavia_ds.tif').transpose(1,2,0))[:,:340]
    y = np.int32(read(data_path+'PaviaU_gt.mat')['paviaU_gt']).ravel()
    source_bands, source_fwhm = readENVIHeader(data_path + 'University.hdr')
    source_bands *= 1e3
    source_fwhm *= 1e3
    num_classes = 9
    padding = ((3,3),(2,2),(0,0))

elif dataset == 'ip':
    data_path = root + 'datasets/indian_pines/'
    X = read(data_path + 'indianpines_ds.tif').transpose(1,2,0)
    X = np.float32(X)
    y = read(data_path + 'indianPinesGT.mat')['indian_pines_gt'] 
    y = y.ravel()
    source_bands, source_fwhm = readENVIHeader(data_path + 'indianpines_ds_raw.hdr')
    num_classes = 16
    padding = ((4,3),(4,3),(0,0))
    
else:
    raise ValueError('Invalid dataset %s: Valid entries include "ip", "paviau"')


oa_arr = np.zeros((num_trials, ))
aa_arr = np.zeros((num_trials, ))
kappa_arr = np.zeros((num_trials, ))

for i in range(num_trials):

    #Create a random training/validation split
    y_train, y_val = random_data_split(y, train_samples=50, shuffle=True)
    
    #Load Spatial-Spectral Feature Extractor
    fe = []
    target_bands = []
    target_fwhm = []
    fe_sensors = ['hyperion'] # This can include 'aviris', 'hyperion', and 'gliht'
    
    for ds in fe_sensors: 
        mat = read(fe_path + '/%s/%s_bands.mat' % (ds,fe_type))
        fe.append(SCAE(ckpt_path=fe_path + '/%s' % ds, nb_cae=3))
        target_bands.append(mat['bands'][0])
        target_fwhm.append(mat['fwhm'][0])
    
    #This pre-processes the data prior to passing it to a feature extractor
    #This can be a string or a ImagePreprocessor objectb
    pre_processor = ['StandardScaler']
    
    #This pre-process the extracted features prior to passing it to a classifier
    #This can be a string or a ImagePreprocessor object
    feature_scaler = [AveragePooling2D(), 'StandardScaler', 'MinMaxScaler',
                      ImagePreprocessor('PCA', whiten=False, PCA_components=0.999)]
    
    #Classifier - SVM-RBF 
    # The SVM from svm_cv_workflow.py does NOT use the validation set - it does
    # N-fold cross-validation with the training set.  This is the ONLY classifier
    # we have that does this.
    clf = SVMWorkflow(cv=7)
    
    #The classification pipeline
    pipe = Pipeline(pre_processor=pre_processor, 
                    feature_extractor=fe,
                    feature_spatial_padding = padding, 
                    feature_scaler=feature_scaler, 
                    classifier=clf, source_bands=source_bands, 
                    source_fwhm=source_fwhm,
                    target_bands=target_bands, target_fwhm=target_fwhm)
    
    #Fit Data
    with Timer('Pipeline Fit Timer'):
        pipe.fit(X, y_train, X, y_val)
    
    #Generate Prediction
    with Timer('Pipeline Predict Timer'):
        pred = pipe.predict(X)
    
    #Evaluate and Print Results
    m = Metrics(y[y>0]-1, pred[y>0])
    oa_arr[i], aa_arr[i], kappa_arr[i] = m.standard_metrics()  

print('Overall Accuracy: %1.2f+/-%1.2f' % (np.mean(oa_arr) * 100.0, np.std(oa_arr)*100.0))
print('Mean-Class Accuracy: %1.2f+/-%1.2f' % (np.mean(aa_arr) * 100.0, np.std(aa_arr)*100.0))
print('Kappa Statistic: %1.4f+/-%1.4f' % (np.mean(kappa_arr), np.std(kappa_arr)))

#SCAE-Hyperion 99.9% 
#Overall Accuracy: 95.25+/-1.37
#Mean-Class Accuracy: 97.05+/-0.54
#Kappa Statistic: 0.9378+/-0.0175