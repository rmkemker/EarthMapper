"""
Name: aviris.py
Author: Ronald Kemker
Description: Pipeline for processing AVIRIS imagery

Note:
Requires GDAL, remote_sensing.utils, and display (optional)
http://www.gdal.org/
"""

import gdal
from glob import glob
import numpy as np
from utils.utils import readENVIHeader, band_resample_hsi_cube

class AVIRIS():
    """
    AVIRIS
    
    This processes data from the Airborne Visible/Infrared Imaging Spectrometer
	(AVIRIS).  
    
    Parameters
    ----------
    directory : string, file directory of AVIRIS data
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
	Ref: https://aviris.jpl.nasa.gov/
    
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
        f = glob(self.dName+'/*ort_img')[0]
        self.data = gdal.Open(f).ReadAsArray().transpose(1,2,0)
        self.shape = self.data.shape
        self.mask = self.data[:,:,10] != self._mask_value()
        self.data[self.mask==False] = 0
        self.hdr_data()
        
        if self.cal:
            self.radiometric_cal()
        
        if self.tgt_bands is not None:
            self.band_resample()
            
    def radiometric_cal(self):
        """Performs radiometric calibration with gain file"""
        with open(glob(self.dName+'/*gain')[0]) as f:
            lines = f.read().splitlines()
            
        gain = np.zeros(224, dtype=np.float32)
        for i in range(len(lines)):
            gain[i] = np.float32(lines[i].split()[0])
        
        self.data[self.mask] = np.float32(self.data[self.mask])*gain
    
    def show_rgb(self):
        """Displays RGB visualization of HSI cube"""
        from display import imshow
        idx = np.array([13,19,27], dtype=np.uint8)
        imshow(self.data[:,:,idx])
    
    def hdr_data(self):
        """Finds band-centers and full-width half-maxes of hyperspectral cube"""
        f = glob(self.dName+'/*ort_img.hdr')[0]        
        self.bands, self.fwhm = readENVIHeader(f)
        
    def band_resample(self):
        """Performs band resampling operation"""
        self.data = band_resample_hsi_cube(self.data,self.bands,self.tgt_bands,
                                           self.fwhm, self.tgt_fwhm,
                                           self.mask)
        
    def _mask_value(self):
        """Finds value of invalid orthomosaic pixels"""
        return np.min([self.data[0,-1],self.data[-1,0],self.data[0,-1],
                       self.data[-1,-1]])
