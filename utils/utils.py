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
import os

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


def set_gpu(device=None):
    
    if device is None:
        device=""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
