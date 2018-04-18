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

class GLiHT():
    """
    GLiHT
    
    This processes hyperspectral data from the G-LiHT:  
        Goddard's LiDAR, Hyperspectral & Thermal Imager
	(GLiHT).  
    
    Parameters
    ----------
    directory : string, file directory of GLiHT data
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
	Ref: https://gliht.gsfc.nasa.gov/
    
    """      
    def __init__(self , directory, calibrate = True, bands=None, fwhm=None):
        self.dName = directory
        self.cal = calibrate
        self.tgt_bands = bands
        self.tgt_fwhm = fwhm
        self.read()
        pass
    
    def read(self):
        '''Reads orthomosaic, calculates binary mask, performs radiometric
        ibration (optional), and band resamples (optional)'''
        f = glob(self.dName+'/*radiance_L1G')[0]
        self.data = gdal.Open(f).ReadAsArray().transpose(1,2,0)
        self.shape = self.data.shape
        self.mask = self.data[:,:,10] != 0
        self.hdr_data()
        
        if self.cal:
            self.radiometric_cal()
        
        if self.tgt_bands is not None:
            self.band_resample()
            
        self.data[self.mask==False] = 0

            
    def radiometric_cal(self):
        """Performs radiometric calibration with gain file"""     
        gain = 1e-4 
        self.data[self.mask] = np.float32(self.data[self.mask])*gain
    
    def show_rgb(self):
        '''Displays RGB visualization of HSI cube'''
        from display import imshow
        idx = np.array([15,33,54], dtype=np.uint8) #blue, green, red
        imshow(self.data[:,:,idx])
    
    def hdr_data(self):
        """Finds band-centers and full-width half-maxes of hyperspectral cube"""
        f = glob(self.dName+'/*radiance_L1G.hdr')[0]        
        self.bands, self.fwhm = readENVIHeader(f)
        
    def band_resample(self):
        """Performs band resampling operation"""
        self.data = band_resample_hsi_cube(self.data,self.bands,self.tgt_bands,
                                           self.fwhm, self.tgt_fwhm)
        
