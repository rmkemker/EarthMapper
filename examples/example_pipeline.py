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
from preprocessing import ImagePreprocessor
from utils.train_val_split import train_val_split
from feature_extraction.scae import SCAE
from classifier.svm_workflow import SVM_Workflow
from classifier.rf_workflow import RandomForestWorkflow

gpu_id = "0"
dataset = 'ip' # dataset choices for this example are 'ip' and 'paviau'
fe_type = 'scae' #feature extractor options are 'scae' and 'smcae'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
root = '/home/rmk6217/Documents/EarthMapper/'
fe_path = root + 'feature_extraction/pretrained/' + fe_type
data_path = root + 'datasets/indian_pines/'

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


#Create a random training/validation split
y_train, y_val = train_val_split(y, 15 , 35)

#Load Spatial-Spectral Feature Extractor
fe = []
target_bands = []
target_fwhm = []
fe_sensors = ['aviris'] # This can include 'aviris', 'hyperion', and 'gliht'

for ds in fe_sensors: 
    mat = read(fe_path + '/%s/%s_bands.mat' % (ds,fe_type))
    fe.append(SCAE(ckpt_path=fe_path + '/%s' % ds, nb_cae=5))
    target_bands.append(mat['bands'][0])
    target_fwhm.append(mat['fwhm'][0])

#This pre-processes the data prior to passing it to a feature extractor
#This can be a string or a ImagePreprocessor object
pre_processor = ['StandardScaler', 'MinMaxScaler']

#This pre-process the extracted features prior to passing it to a classifier
#This can be a string or a ImagePreprocessor object
feature_scaler = ['StandardScaler', 'MinMaxScaler', 
                  ImagePreprocessor(mode='PCA', PCA_components=0.99)]

#Classifier - SVM-RBF (probability=True required for CRF post-processor)
#clf = SVM_Workflow(probability=True, kernel='rbf')
clf = RandomForestWorkflow()

#The classification pipeline
pipe = Pipeline(pre_processor=pre_processor, 
                feature_extractor=fe,
                feature_spatial_padding = padding, 
                feature_scaler=feature_scaler, 
                classifier=clf, source_bands=source_bands, 
                source_fwhm=source_fwhm,
                target_bands=target_bands, target_fwhm=target_fwhm,
                post_processor='CRF')

#Fit Data
with Timer('Pipeline Fit Timer'):
    pipe.fit(X , y_train, X, y_val)

#Generate Prediction
with Timer('Pipeline Predict Timer'):
    pred = pipe.predict(X)

#Evaluate and Print Results
m = Metrics(y[y>0]-1, pred[y>0])
overall_accuracy, mean_class_accuracy, kappa_statistic = m.standard_metrics()  
confusion_matrix = m.c 
print('Overall Accuracy: %1.2f%%' % (overall_accuracy * 100.0))
print('Mean-Class Accuracy: %1.2f%%' % (mean_class_accuracy * 100.0))
print('Kappa Statistic: %1.4f' % (kappa_statistic))


    
