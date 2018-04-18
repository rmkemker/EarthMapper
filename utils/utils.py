"""
Name: utils.py
Author: Ronald Kemker
Description: Helper functions for remote sensing applications.

Note:
Requires SpectralPython and GDAL
http://www.spectralpython.net/
http://www.gdal.org/
"""

import numpy as np
from spectral import BandResampler

import sys, os

def band_resample_hsi_cube(data, bands1, bands2, fwhm1,fwhm2 , mask=None):
    """
    band_resample_hsi_cube : Resample hyperspectral image

	Parameters
	----------    
	data : numpy array (height x width x spectral bands)
    bands1 : numpy array [1 x num source bands], 
		the band centers of the input hyperspectral cube
    bands2 : numpy array [1 x num target bands], 
		the band centers of the output hyperspectral cube
    fwhm1  : numpy array [1 x num source bands], 
		the full-width half-max of the input hyperspectral cube
    fwhm2 : numpy array [1 x num target bands],
		the full-width half-max of the output hyperspectral cube
	mask : numpy array (height x width), optional mask to perform the band-
		resampling operation on.

    Returns
    -------
	output - numpy array (height x width x N)

    """
    resample = BandResampler(bands1,bands2,fwhm1,fwhm2)
    dataSize = data.shape
    data = data.reshape((-1,dataSize[2]))
    
    if mask is None:
        out = resample.matrix.dot(data.T).T
    else:
        out = np.zeros((data.shape[0], len(bands2)), dtype=data.dtype)
        mask = mask.ravel()
        out[mask] = resample.matrix.dot(data[mask].T).T
        
    out[np.isnan(out)] = 0
    return out.reshape((dataSize[0],dataSize[1],len(bands2)))

    
from sklearn.model_selection import train_test_split

def class_weights(labels, mu , numClasses=None):
    if numClasses is None:
        numClasses= np.max(labels)+1    
    class_weights = np.bincount(labels, minlength=numClasses)       
    class_weights = np.sum(class_weights)/class_weights
    class_weights[np.isinf(class_weights)] = 0
    class_weights = mu * np.log(class_weights)
    class_weights[class_weights < 1] = 1
    return class_weights


def train_test_split_per_class(X, y, train_size=None, test_size=None):
    
    sh = np.array(X.shape)
    
    num_classes = len(np.bincount(y))
    
    sh[0] = 0
    X_train_arr =  np.zeros(sh, dtype=X.dtype)
    X_test_arr = np.zeros(sh, dtype=X.dtype)
    y_train_arr = np.zeros((0), dtype=y.dtype)
    y_test_arr = np.zeros((0), dtype=y.dtype)

    for i in range(num_classes):
        X_train, X_test, y_train, y_test = train_test_split(X[y==i], y[y==i],
                                                            train_size=train_size,
                                                            test_size=test_size)
        
        X_train_arr =  np.append(X_train_arr, X_train, axis=0)
        X_test_arr = np.append(X_test_arr, X_test, axis=0)
        y_train_arr = np.append(y_train_arr, y_train)
        y_test_arr = np.append(y_test_arr, y_test)
        
    return X_train_arr, X_test_arr, y_train_arr, y_test_arr

def set_gpu(device=None):
    
    if device is None:
        device=""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
