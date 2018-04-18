"""
Name: hyperion.py
Author: Ronald Kemker
Description: Pipeline for processing Hyperion imagery

Note:
Requires GDAL, remote_sensing.utils, and display (optional)
http://www.gdal.org/
"""

import gdal
from glob import glob
import numpy as np
from utils.utils import band_resample_hsi_cube

class Hyperion():
    """
    Hyperion
    
    This processes data from the Hyperion HSI sensor.
	(Hyperion).  
    
    Parameters
    ----------
    directory : string, file directory of Hyperion data
	calibrate : boolean, convert hyperspectral cube frol digital count to 
		radiance.  (Default: True)
	bands : numpy array [1 x num target bands], target bands for spectral 
		resampling (Default: None -> No resampling is performed)
	fwhm : numpy array [1 x num target bands], target full-width half maxes for
		for spectral resampling (Default: None)
        
    Attributes
    ----------
    dName : string, input "directory"
    cal : boolean, input "calibrate"
	bands : numpy array [1 x num source bands], band centers of input cube
	fwhm : numpy array [1 x num source bands], full width, half maxes of input 
		cube
	tgt_bands : numpy array [1 x num target bands], input "bands"
	tgt_fwhm : numpy array [1 x num target bands], input "fwhm"
	data : numpy array, hyperspectral cube
	shape : tuple, shape of hyperspectral cube
    mask : numpy array, binary mask of valid data (1 is valid pixel)

    Notes
    -----
	Ref: https://eo1.usgs.gov/sensors/hyperion
    
    """      
    def __init__(self , directory, calibrate = True, bands=None, fwhm=None):
        self.dName = directory
        self.cal = calibrate
        self.tgt_bands = bands
        self.tgt_fwhm = fwhm
        self.read()
        pass
    
    def read(self):
        """Reads orthomosaic, calculates binary mask, performs radiometric
		calibration (optional), and band resamples (optional)"""       
        dataFN = sorted(glob(self.dName+'/*.TIF'))
        self.calIdx = np.hstack([np.arange(7,57),np.arange(76,224)])             

        tmp = gdal.Open(dataFN[self.calIdx[0]]).ReadAsArray()
        
        self.data = np.zeros((tmp.shape[0],tmp.shape[1],198),dtype=np.float32) 
        self.shape = self.data.shape
        self.hdr_data()

        self.data[:,:,0] = tmp
        for i in range(1,198):
            self.data[:,:,i] = gdal.Open(dataFN[self.calIdx[i]]).ReadAsArray()
              
        if self.cal:
           self.data[:,:,0:50] = self.data[:,:,0:50]/40
           self.data[:,:,50:] = self.data[:,:,50:]/80  

        self.mask = self.data[:,:,10] != self._mask_value()

        if self.tgt_bands is not None:
            self.band_resample()
        
        self.data[self.mask==False] = 0

    def show_rgb(self):
        """Displays RGB visualization of HSI cube"""
        from display import imshow
        idx = np.array([20,10,0], dtype=np.uint8)
        imshow(self.data[:,:,idx])
    
    def hdr_data(self):
        """Finds band-centers and full-width half-maxes of hyperspectral cube"""
        hdr = np.genfromtxt('/home/rmk6217/Documents/grss_challenge/fileIO/hyperion_bands.csv', delimiter=',')
        self.bands = hdr[:,0]
        self.fwhm = hdr[:,1]
        
               
    def band_resample(self):
        """Performs band resampling operation"""
        self.data = band_resample_hsi_cube(self.data,self.bands,self.tgt_bands,
                                           self.fwhm, self.tgt_fwhm)
        
    def _mask_value(self):
        """Finds value of invalid orthomosaic pixels"""
        return np.min([self.data[0,-1],self.data[-1,0],self.data[0,-1],
                       self.data[-1,-1]])

    
    