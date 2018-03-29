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
import subprocess

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

def bash(bash_cmd):
    """Bash command operation
    
    Parameters
    ----------
    bash_cmd : string, the bash command to be sent to the terminal

    Returns
    -------
    output : output of the input command
    """  
    process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
    return process.communicate()
