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
    
class ZCAWhiten():
    """
    Class used to scale data using ZCA whitening by fitting and transforming
    Args:
        epsilon (float): Default value is 0.1
    """
    def __init__(self, epsilon=0.1):
        self.eps = epsilon
        
    def fit(self, X):       
        """
        Sets the scale based on the input data using StandardScaler
        Args:
            X (array): the data to be used to set the scale
        """
        self.sclr = StandardScaler(with_std=False)
        X = self.sclr.fit_transform(X)
                
        X = X.T
        sigma = np.dot(X, X.T)/X.shape[-1] 
        U,S,V = np.linalg.svd(sigma) 
        self.ZCAMatrix = np.dot(np.dot(U, 
                            1.0/np.sqrt(np.diag(S) + self.eps)), U.T)                                    
    def transform(self, X):
        """
        Transforms data based on the ZCAMatrix and the scale set in the fit 
            function
        Args:
            X (array): the data to be transformed according to the scale
        Returns: the transformed data
        """
        X = self.sclr.transform(X)
        X = X.T
        return np.dot(self.ZCAMatrix, X).T
    
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
    def __init__(self, ord=None):
        self.ord = ord
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
    
    def __init__(self, pool_size=5, large=False):
        self.pool_size = pool_size
        self.large = large
    def fit(self, X):
        pass
    def transform(self, X):
        return mean_pooling(X, self.pool_size)
    def fit_transform(self , X):
        return self.transform(X)   

class GlobalContrastNormalization():
    """
    Class used to scale data by fitting and transforming it
    Args:
        sqrt_bias (int): Default value is 10
        epsilon (float): Default value is 1e-8
        with_std (boolean): Default value is True
        scale (int): Default value is 1
    """
    def __init__(self, sqrt_bias = 10, epsilon = 1e-8, with_std = True, scale=1):
        self.sqrt_bias = sqrt_bias
        self.eps= epsilon
        self.scale = scale
        self.with_std = with_std
        
    def fit(self, X):
        """
        Sets the scale based on the input data
        Args:
            X (array): the data to be used to set the scale
        """
        assert X.ndim == 2, "X.ndim must be 2"
        self.scale = float(self.scale)
        assert self.scale >= self.eps
        
        self.mean = X.mean(axis=1)
        X = X - self.mean[:, np.newaxis]
        
        if self.with_std:
            ddof = 1
            if X.shape[1] == 1:
                ddof = 0
            self.normalizers = np.sqrt(self.sqrt_bias + X.var(axis=1, ddof=ddof)) / self.scale
        else:
            self.normalizers = np.sqrt(self.sqrt_bias + (X ** 2).sum(axis=1)) / self.scale
        
        self.normalizers[self.normalizers < self.eps] = 1.
        
    def transform(self, X):
        """
        Transforms data based on the scale set in the fit function
        Args:
            X (array): the data to be transformed according to the scale
        Returns: the transformed data
        """
        X = X - self.mean[:, np.newaxis]
        return X/self.normalizers[:, np.newaxis]
    
    def fit_transform(self , X):
        """
        Performs both the fit and the transform methods on the same input data
        Args:
            X (array): the data to be used to set the scale then transformed
        Returns: the result of transforming the X data after fitting it
        """
        self.fit(X)
        return self.transform(X)
        

class ImagePreprocessor():
    
    def __init__(self, mode='StandardScaler' , feature_range=[0,1], 
                 with_std=True, PCA_components=None, svd_solver='auto',
                 ord=None, eps=0.05, whiten=True):
        self.mode = mode.lower()
        self.with_std = with_std
        if self.mode == 'standardscaler':
            self.sclr = StandardScaler(with_std=with_std)
        elif self.mode == 'minmaxscaler':
            self.sclr = MinMaxScaler(feature_range = feature_range)
        elif self.mode == 'pca':
            self.sclr = PCA(n_components=PCA_components,whiten=whiten, 
                            svd_solver=svd_solver)
        elif self.mode == 'globalcontrastnormalization':
            self.sclr = GlobalContrastNormalization(with_std=with_std)
        elif self.mode == 'zcawhiten':
            self.sclr = ZCAWhiten()
        elif self.mode == 'normalize':
            self.sclr = Normalize()
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
        elif self.mode == 'globalcontrastnormalization':
            return self.sclr.mean, self.sclr.normalizers
        elif self.mode == 'zcawhiten':
            return self.sclr.ZCAMatrix
        elif self.mode == 'normalize':
            return self.sclr.ord
        elif self.mode == 'exponential':
            return self.sclr.scale
        else:
            return None
           
if __name__ == '__main__':   
    sclr = ImagePreprocessor('Exponential')
    x = np.random.rand(100, 32,32, 3) * 255
    sclr.fit(x)
    f = sclr.transform(x)
    f2 = sclr.fit_transform(x)
    scale = sclr.get_params()
