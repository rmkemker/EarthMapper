"""
Name: preprocessing.py
Author: Ronald Kemker
Description: Image Pre-processing workflow
Note:
Requires scikit-learn
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from utils.convolution import mean_pooling
import numpy as np
from scipy.stats import expon

class ConeResponse():
    def __init__(self, eps=0.05):
        self.eps = eps
    def fit(self, X):
        pass
    def transform(self, X):
        return (np.log(self.eps)-np.log(X + self.eps))/(np.log(self.eps)-np.log(1 + self.eps))
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class ReLU():
    def __init__(self):
        pass
    def fit(self, X):
        pass
    def transform(self, X):
        X[X<0] = 0
        return X
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class Exponential():
    """
    Class used to exponentially scale data by fitting and transforming it
    """
    def __init__(self):
        pass    
    def fit(self, X):
        """
        Sets the scale based on the input data
        Args:
            X (array): the data to be used to set the scale
        """
        self.scale = np.zeros(X.shape[-1], dtype=np.float32)
        for i in range(0,X.shape[-1]):
            _, self.scale[i] = expon.fit(X[:,i],floc=0)
    def transform(self, X):
        """
        Transforms data based on the scale set in the fit function
        Args:
            X (array): the data to be transformed according to the scale
        Returns: the transformed data
        """
        return 1-np.exp(-np.abs(X)/self.scale)
    def fit_transform(self, X):
        """
        Performs both the fit and the transform methods on the same input data
        Args:
            X (array): the data to be used to set the scale then transformed
        Returns: the result of transforming the X data after fitting it
        """
        self.fit(X)
        return self.transform(X)
    
class Normalize():
    """
    Class used to normalize and scale data by fitting and transforming it
    Args:
        ord: Default value is None
    """
    def __init__(self, norm_order=None):
        self.ord = norm_order
    def fit(self, X):
        """
        Sets the scale based on the input data; uses np.linalg.norm
        Args:
            X (array): the data to be used to set the scale
        """
        self.ord = np.linalg.norm(X, axis=0)
        self.ord[self.ord==0] = 1e-7
    def transform(self, X):
        """
        Transforms data based on the scale set in the fit function
        Args:
            X (array): the data to be transformed according to the scale
        Returns: the transformed data
        """
        return X/self.ord
    def fit_transform(self , X):
        """
        Performs both the fit and the transform methods on the same input data
        Args:
            X (array): the data to be used to set the scale then transformed
        Returns: the result of transforming the X data after fitting it
        """
        self.fit(X)
        return self.transform(X)

class AveragePooling2D():
    
    def __init__(self, pool_size=5):
        self.pool_size = pool_size
    def fit(self, X):
        pass
    def transform(self, X):
        return mean_pooling(X, self.pool_size)
    def fit_transform(self , X):
        return self.transform(X)   

class ImagePreprocessor():
    
    def __init__(self, mode='StandardScaler' , feature_range=[0,1], 
                 with_std=True, PCA_components=None, svd_solver='auto',
                 norm_order=None, eps=0.05, whiten=True):
        self.mode = mode.lower()
        self.with_std = with_std
        if self.mode == 'standardscaler':
            self.sclr = StandardScaler(with_std=with_std)
        elif self.mode == 'minmaxscaler':
            self.sclr = MinMaxScaler(feature_range = feature_range)
        elif self.mode == 'pca':
            self.sclr = PCA(n_components=PCA_components,whiten=whiten, 
                            svd_solver=svd_solver)
        elif self.mode == 'normalize':
            self.sclr = Normalize(norm_order)
        elif self.mode == 'exponential':
            self.sclr = Exponential()
        elif self.mode == 'coneresponse':
            self.sclr = ConeResponse(eps)
        elif self.mode == 'relu':
            self.sclr = ReLU()
        else:
            raise ValueError('Invalid pre-processing mode: ' + mode)
            
    def fit(self, data):
        data = data.reshape(-1,data.shape[-1])
        self.sclr.fit(data)
    def transform(self,data):
        sh = data.shape
        data = data.reshape(-1,sh[-1])
        data = self.sclr.transform(data)
        return data.reshape(sh[:-1] + (data.shape[-1],))
    def fit_transform(self, data):
        sh = data.shape
        data = data.reshape(-1,sh[-1])
        data = self.sclr.fit_transform(data)
        return data.reshape(sh[:-1] + (data.shape[-1],))
    def get_params(self):
        if self.mode == 'standardscaler':
            if self.with_std:
                return self.sclr.mean_ , self.sclr.scale_
            else:
                return self.sclr.mean_
        elif self.mode == 'minmaxscaler':
            return self.sclr.data_min_, self.sclr.data_max_
        elif self.mode == 'pca':
            return self.sclr.components_
        elif self.mode == 'normalize':
            return self.sclr.ord
        elif self.mode == 'exponential':
            return self.sclr.scale
        else:
            return None