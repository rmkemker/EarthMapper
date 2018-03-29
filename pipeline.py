#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:22:36 2017

@author: Ronald Kemker and Utsav Gewali
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from preprocessing import ImagePreprocessor
from postprocessing import MRF, CRF
from utils.utils import band_resample_hsi_cube

class DummyOp(object):
    def __init__(self):
        pass
    
    def transform(self, X):
        return X

class Pipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline to classify single-image hyperspectral image cube
     - Deployable Framework for Remote Sensing Classification

    Note: TODO Link to paper here

    Args:
        classifier (obj): A classifier object.  
        pre_processor (str, list, obj, optional): Pre-process the image data.  
            It can either be a string ('MinMaxScaler', 'StandardScaler', 
            'PCA', 'Normalize','Exponential', 'ConeResponse, or 'ReLU'), an 
            ImagePreprocessor object, or a list containing one or more of the
            previous strings or ImagePre-processor objects.
        feature_extractor (obj, optional): A spatial-spectral feature extractor
            object (e.g. MICA or SCAE).  
        feature_scaler (str, list, obj, optional): If a feature extractor is
            used, you can pre-process the data again prior to classification.
            The input format is the same as the pre_processor object.
        feature_spatial_padding (2-D tuple): This is the pad_width argument to
            numpy.pad.  This is used if spatial padding is required for feature
            extraction.
        post_processor (str, obj, optional): The post-processing object or 
            string with the name of that post-processing object.  The only 
            supported post-processor at this time is 'CRF'.
        source_bands (array of floats, optional): The band centers of the  
            source data.
        source_fwhm (array of floats, optional): The full-width, half maxes   
            of the source data.  Should be the same shape as source_bands.
        target_bands (array of floats, optional): The band centers of the  
            target data.
        target_fwhm (array of floats, optional): The full-width, half maxes   
            of the target data.  Should be the same shape as target_bands.
        semisupervised (boolean, optional): If true, the classifier will treat
            all labels = -1 as unlabeled training data. Labels < -1 are
            ignored. (Default: False)
        **kwargs (optional): These are various input variables to the 
            pre-processor, feature_scaler, classifier, and post_processor 
            class instances.  This doesn't really work well...
            
    Attributes:
        
    """
    def __init__(self, classifier, pre_processor=None, feature_extractor=None, 
                 feature_scaler=[], feature_spatial_padding=None,
                 post_processor=None, 
                 source_bands=[], source_fwhm=[], target_bands=[], 
                 target_fwhm=[], layer_sizes=None, denoising_cost=None,
                 semisupervised = False, **kwargs):

        valid_pre = ['MinMaxScaler', 'StandardScaler', 'PCA',
                     'GlobalContrastNormalization', 'ZCAWhiten', 'Normalize',
                     'Exponential','ReLU','ConeResponse', 'AveragePooling2D']
        
        valid_post = ['MRF','CRF']

        self.classifier = classifier
        self.semisupervised = semisupervised        
           
        self.probability = post_processor is not None
        kwargs['probability'] = self.probability
        
        #Pre-process the data
        if pre_processor is None:
            pre_processor = []
        elif type(pre_processor) is not list:
            pre_processor = [pre_processor]
        
        self.pre_processor = []
        for pre in pre_processor: 
            if type(pre) == str:
                if pre in valid_pre:
                    self.pre_processor.append(ImagePreprocessor(pre))
                    for k in kwargs.keys():
                        if k in self.pre_processor[-1].__dict__.keys():
                            self.pre_processor[-1].__dict__[k] = kwargs[k]
                else:
                    msg = 'Invalid pre_processor type: %s. Built-in options include %s.'
                    raise ValueError(msg % (pre, str(valid_pre)[1:-1]))
            else:
                self.pre_processor.append(pre)


        #Pre-process the data after feature extraction
        if type(feature_scaler) is not list:
            feature_scaler = [feature_scaler]
        
        self.feature_scaler = []
        for pre in feature_scaler: 
            if type(pre) == str:
                if pre in valid_pre:
                    self.feature_scaler.append(ImagePreprocessor(pre))
                    for k in kwargs.keys():
                        if k in self.feature_scaler[-1].__dict__.keys():
                            self.feature_scaler[-1].__dict__[k] = kwargs[k]
                else:
                    msg = 'Invalid pre_processor type: %s. Built-in options include %s.'
                    raise ValueError(msg % (pre, str(valid_pre)[1:-1]))
            else:
                self.feature_scaler.append(pre)

        if type(post_processor) == str:
            if post_processor in valid_post:
                self.post_processor = eval(post_processor)()
            else:
                msg = 'Invalid post-processor type: %s. Built-in options include %s.'
                raise ValueError(msg % (post_processor, str(valid_post)[1:-1]))
        else:
            self.post_processor = post_processor
            
        self.feature_extractor = feature_extractor
        self.source_bands = source_bands
        self.source_fwhm = source_fwhm
        self.target_bands = target_bands
        self.target_fwhm = target_fwhm
        self.pad = feature_spatial_padding
        
    def fit(self, X_train, y_train, X_val, y_val):
        
        """
        Fits to training data.
        
        Args:
            X_train (ndarray): the X training data
            y_train (ndarray): the y training data
            X_val (ndarray): the X validation data
            y_val (ndarray): the y validation data
        """
        
        same = np.all(X_train == X_val)
        
        #Data pre-processing        
        for i in range(len(self.pre_processor)):
            self.pre_processor[i].fit(X_train)
            X_train = self.pre_processor[i].transform(X_train)
            if not same:
                X_val = self.pre_processor[i].transform(X_val)
        
        if self.pad:
            X_train = np.pad(X_train, self.pad, mode='constant')
            if not same:
                X_val = np.pad(X_val, self.pad, mode='constant')
        
        if not self.feature_extractor:
            self.feature_extractor = [DummyOp()]
            self.target_bands= [[]]
            self.target_fwhm = [[]]
        elif not isinstance(self.feature_extractor, list):
            self.feature_extractor = [self.feature_extractor]
            self.target_bands = [self.target_bands]
            self.target_fwhm = [self.target_fwhm]
            
        base_train = np.copy(X_train)
        final_train = np.zeros((base_train.shape[:2] + (0, )),dtype=np.float32)
        if not same:
            base_val = np.copy(X_val)
            final_val = np.zeros((base_val.shape[:2] + (0, )),dtype=np.float32)
        
        for i, fe in enumerate(self.feature_extractor):
            #Band resampling
            if len(self.target_bands[i]) and len(self.target_fwhm[i]) and \
                len(self.source_bands) and len(self.source_fwhm):
                X_train = band_resample_hsi_cube(base_train, self.source_bands, 
                                                          self.target_bands[i], 
                                                          self.source_fwhm, 
                                                          self.target_fwhm[i])
                if not same:
                    X_val = band_resample_hsi_cube(base_val, self.source_bands, 
                                                          self.target_bands[i], 
                                                          self.source_fwhm, 
                                                          self.target_fwhm[i])  
            #Feature extraction
            X_train = fe.transform(X_train)
            if not same:               
                X_val = fe.transform(X_val)

            final_train = np.append(final_train, X_train, axis=-1)
            if not same:
                final_val = np.append(final_val, X_val, axis=-1)
        
        
        X_train = np.copy(final_train)
        del base_train, final_train
        if not same:
            X_val = np.copy(final_val)
            del base_val, final_val
        
        if self.pad:
            X_train = X_train[self.pad[0][0]:-self.pad[0][1],
                              self.pad[1][0]:-self.pad[1][1]]
            if not same:
                X_val = X_val[self.pad[0][0]:-self.pad[0][1],
                              self.pad[1][0]:-self.pad[1][1]]                    


        print(X_train.shape)
        if len(self.feature_scaler):
            for feature_scaler_i in self.feature_scaler:
                feature_scaler_i.fit(X_train) #TODO: Combine train and val
                X_train = feature_scaler_i.transform(X_train)
                if not same:
                    X_val = feature_scaler_i.transform(X_val)            
        
        if same:
            X_val = X_train
        
        if self.post_processor:
            feature2d = X_val
            val_idx = y_val.ravel() > -1
        
        #Reshape to N x F
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(-1, X_train.shape[-1])
        if len(X_val.shape) > 2:
            X_val = X_val.reshape(-1, X_val.shape[-1])
        if len(y_train.shape) > 1:
            y_train = y_train.ravel()
        if len(y_val.shape) > 1:
            y_val = y_val.ravel()
        
        if not self.semisupervised:
            X_train = X_train[y_train > -1]
            y_train = y_train[y_train > -1]
        else:
            X_train = X_train[y_train >= -1]
            y_train = y_train[y_train >= -1]
        
        X_val = X_val[y_val > -1]
        y_val = y_val[y_val > -1]
        
        print("Train: %d x %d" % (X_train.shape[0], X_train.shape[1]))
        print("Val: %d x %d" % (X_val.shape[0], X_val.shape[1]))
                
        #Fit classifier
        self.classifier.fit(X_train, y_train, X_val, y_val)
        
        #Fit MRF/CRF
        if self.post_processor:
            sh = feature2d.shape
            prob_map = self.classifier.predict_proba(feature2d.reshape(-1, sh[-1]))
            prob_map = prob_map.reshape(sh[0], sh[1], -1)
            self.post_processor.fit(prob_map+1e-50, y_val, val_idx)
            
                    
    def predict(self, X):
        
        """
        Makes predictions based on training and X.
        
        Args:
            X (ndarray): the testing data
        
        Returns:
            pred (ndarray): the predictions of the y data based 
                on training and X
        """

        if len(self.pre_processor):
            for j in range(len(self.pre_processor)):
                X = self.pre_processor[j].transform(X)

        if self.pad:
            X = np.pad(X, self.pad, mode='constant')
        
        base = np.copy(X)
        final = np.zeros(X.shape[:2] + (0,),dtype=np.float32)
        
        for i, fe in enumerate(self.feature_extractor):
            if len(self.target_bands[i]) and len(self.target_fwhm[i])  and \
                len(self.source_bands) and len(self.source_fwhm):
                X = band_resample_hsi_cube(base, self.source_bands, 
                                              self.target_bands[i], 
                                              self.source_fwhm, 
                                              self.target_fwhm[i])
            else:
                X = np.copy(base)
                
            #Feature extraction
            final = np.append(final,  fe.transform(X) , axis=-1)
         
            
        X = np.copy(final)
        del base, final
        
        if self.pad:
            #TODO: generalize this to N-dimensions
            X = X[self.pad[0][0]:-self.pad[0][1],
                  self.pad[1][0]:-self.pad[1][1]]            
        
        if len(self.feature_scaler):
            for feature_scaler_i in self.feature_scaler:
                X = feature_scaler_i.transform(X)
        
        
        sh = X.shape
        if len(X.shape) > 2:
            X = X.reshape(-1, X.shape[-1])
                    
        if self.post_processor:
            prob = self.classifier.predict_proba(X)
            pred = self.post_processor.predict(prob.reshape(sh[0], sh[1], -1)+1e-50)
            pred = pred.ravel()
        else:
            pred = self.classifier.predict(X)
        
    
        return pred
    
    def predict_proba(self, X):
        
        """
        Finds probabilities of possible outcomes using samples from X.
        
        Args:
            X (ndarray): the testing data
        
        Returns: ndarray of possible outcomes based on samples from X
        """
        
        if len(self.pre_processor):
            for j in range(len(self.pre_processor)):
                X = self.pre_processor[j].transform(X)

        if self.pad:
            X = np.pad(X, self.pad, mode='constant')
        
        base = np.copy(X)
        final = np.zeros(X.shape[:2] + (0,),dtype=np.float32)
        
        for i, fe in enumerate(self.feature_extractor):
            if len(self.target_bands[i]) and len(self.target_fwhm[i])  and \
                len(self.source_bands) and len(self.source_fwhm):
                X = band_resample_hsi_cube(base, self.source_bands, 
                                              self.target_bands[i], 
                                              self.source_fwhm, 
                                              self.target_fwhm[i])
            else:
                X = np.copy(base)
                
            #Feature extraction
            final = np.append(final,  fe.transform(X) , axis=-1)
         
            
        X = np.copy(final)
        del base, final
        
        if self.pad:
            #TODO: generalize this to N-dimensions
            X = X[self.pad[0][0]:-self.pad[0][1],
                  self.pad[1][0]:-self.pad[1][1]]            
        
        if len(self.feature_scaler):
            for feature_scaler_i in self.feature_scaler:
                X = feature_scaler_i.transform(X)
                
        if len(X.shape) > 2:
            X = X.reshape(-1, X.shape[-1])
                    
        proba = self.classifier.predict_proba(X)
      
        return proba
