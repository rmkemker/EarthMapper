"""
Name: fileIO.py
Author: Ronald Kemker
Description: Input/Output helper functions

Note:
Requires GDAL, remote_sensing.utils, and display (optional)
http://www.gdal.org/
"""

import numpy as np
import pickle
from scipy.io import loadmat, savemat
from gdal import Open
import spectral.io.envi as envi
from numba import jit
from random import sample

def read(fName):
    """File reader
    
    Parameters
    ----------
    fname : string, file path to read in (supports .npy, .sav, .pkl, and 
	.tif (GEOTIFF)) including file extension
    
    Returns
    -------
    output : output data (in whatever format)
    """    
    ext = fName[-3:]
    
    if ext == 'npy':
        return np.load(fName)        
    elif ext == 'sav' or ext == 'pkl':
        return pickle.load(open(fName, 'rb'))
    elif ext == 'mat':
        return loadmat(fName)
    elif ext == 'tif' or 'pix':
        return Open(fName).ReadAsArray()
    else:
        print('Unknown filename extension')
        return -1

def write(fName, data):
    """File writer
    
    Parameters
    ----------
    fname : string, file path to read in (supports .npy, .sav, .pkl, and 
		.tif (GEOTIFF)) including file extension
    data : data to be stored
    Returns
    -------
    output : output data (in whatever format)
    """  
    ext = fName[-3:]
    
    if ext == 'npy':
        np.save(fName, data)  
        return 1
    elif ext == 'sav' or ext == 'pkl':
        pickle.dump(data, open(fName, 'wb'))
        return 1
    elif ext == 'mat':
        savemat(fName, data)
        return 1
    else:
        print('Unknown filename extension')
        return -1

def readENVIHeader(fName):
    """
    Reads envi header

    Parameters
    ----------
    fName : String, Path to .hdr file
    
    Returns
    -------
    centers : Band-centers
	fwhm : full-width half-maxes
    """
    hdr = envi.read_envi_header(fName)
    centers = np.array(hdr['wavelength'],dtype=np.float)
    fwhm = np.array(hdr['fwhm'],dtype=np.float)
    return centers, fwhm
    
@jit
def patch_extractor_with_mask(data, mask, num_patches=50, patch_size=16):
    """
    Extracts patches inside a mask.  I need to find a faster way of doing this.

    Parameters
    ----------
    data : 3-D input numpy array [rows x columns x channels] 
	mask : 2-D binary mask where 1 is valid, 0 else
	num_patches : int, number of patches to extract (Default: 50)
    patch_size : int, pixel dimension of square patch (Default: 16)

    Returns
    -------
    output : 4-D Numpy array [num patches x rows x columns x channels]
	"""    
    sh = data.shape
    patch_arr = np.zeros((num_patches,patch_size, patch_size, sh[-1]), 
                         dtype=data.dtype)
    
    if type(patch_size) == int:
        patch_size = np.array([patch_size, patch_size], dtype = np.int32)
    
    #Find Valid Regions to Extract Patches
    valid = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(0, sh[0]-patch_size[0]):
        for j in range(0, sh[1]-patch_size[1]):
            if np.all(mask[i:i+patch_size[0], j:j+patch_size[1]]) == True:
                valid[i,j] += 1
    
    idx = np.argwhere(valid > 0 )
    idx = idx[np.array(sample(range(idx.shape[0]), num_patches))]
    
    for i in range(num_patches):
        patch_arr[i] = data[idx[i,0]:idx[i,0]+patch_size[0], 
                idx[i,1]:idx[i,1]+patch_size[1]]
        
    return patch_arr    

@jit
def patch_extractor(data, num_patches=50, patch_size=16):
    """
    Extracts patches inside a mask.  I need to find a faster way of doing this.

    Parameters
    ----------
    data : 3-D input numpy array [rows x columns x channels] 
	num_patches : int, number of patches to extract (Default: 50)
    patch_size : int, pixel dimension of square patch (Default: 16)

    Returns
    -------
    output : 4-D Numpy array [num patches x rows x columns x channels]
	"""    
    sh = data.shape
    patch_arr = np.zeros((num_patches,patch_size, patch_size, sh[-1]), 
                         dtype=data.dtype)
    
    if type(patch_size) == int:
        patch_size = np.array([patch_size, patch_size], dtype = np.int32)
    
    #Find Valid Regions to Extract Patches
    valid = np.zeros(data.shape[:2], dtype=np.uint8)
    valid[0: sh[0]-patch_size[0], 0: sh[1]-patch_size[1]] = 1
    
    idx = np.argwhere(valid > 0 )
    idx = idx[np.array(sample(range(idx.shape[0]), num_patches))]
    
    for i in range(num_patches):
        patch_arr[i] = data[idx[i,0]:idx[i,0]+patch_size[0], 
                idx[i,1]:idx[i,1]+patch_size[1]]
    return patch_arr    
