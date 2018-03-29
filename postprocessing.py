"""
Classes for gridwise MRF and dense CRF
Uses pydense (https://github.com/lucasb-eyer/pydensecrf) and python_gco (https://github.com/amueller/gco_python) libraries
"""
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
import pygco as gco
from tqdm import tqdm
from multiprocessing import Process, Queue

class MRF(object):
    """
    MRF postprocessing
    -uses alpha-beta swap algorithm from Multilabel optimization library GCO (http://vision.csd.uwo.ca/code/)
    -unary energy is derived from predicted probability map
    -pairwise energy is Potts function (parameter optimized using grid search)
    """
    
    def __init__(self,inference_niter=50, multiplier=1000):
        self.INFERENCE_NITER = inference_niter #number of iteration of alpha-beta swap
        self.MULTIPLIER = multiplier #GCO uses int32 for energy therefore floating point energy is
                                     #scaled by multiplier and truncated 

  
    def fit(self, X, y, idx, img):#img is unused here but is needed for the pipeline
        self.unaryEnergy = np.ascontiguousarray(-self.MULTIPLIER*np.log(X)).astype('int32') #energy=-log(probability)
        nClass = self.unaryEnergy.shape[2]

        params = np.logspace(-3,3,10) #grid search values for Potts compatibility
        best_score = 0.0

        print('Fitting Grid MRF...')
        for i in tqdm(range(0,params.shape[0])):            
            pairwiseEnergy = (self.MULTIPLIER*params[i]*(1-np.eye(nClass))).astype('int32') 
            out = gco.cut_simple(unary_cost=self.unaryEnergy,\
                  pairwise_cost=pairwiseEnergy,n_iter=self.INFERENCE_NITER,
                  algorithm='swap').ravel()            
            
            score = np.sum(out[idx]==y)/float(y.size)
            if score > best_score:
                self.cost = params[i]
                best_score = score
             
        params = np.logspace(np.log10(self.cost)-1,np.log10(self.cost)+1,30)
        best_score = 0.0
        
        print('Finetuning Grid MRF...')
        for i in tqdm(range(0,params.shape[0])):            
            pairwiseEnergy = (self.MULTIPLIER*params[i]*(1-np.eye(nClass))).astype('int32') 
            out = gco.cut_simple(unary_cost=self.unaryEnergy,\
                  pairwise_cost=pairwiseEnergy,n_iter=self.INFERENCE_NITER,
                  algorithm='swap').ravel()            
            
            score = np.sum(out[idx]==y)/float(y.size)
            if score > best_score:
                self.cost = params[i]
                best_score = score

    def predict(self, X):
        #self.cost = 2. 
        self.unaryEnergy = np.ascontiguousarray(-self.MULTIPLIER*np.log(X)).astype('int32')
        nClass = self.unaryEnergy.shape[2]        
        pairwiseEnergy = (self.MULTIPLIER*self.cost*(1-np.eye(nClass))).astype('int32') 
        return gco.cut_simple(unary_cost=self.unaryEnergy,\
              pairwise_cost=pairwiseEnergy,n_iter=self.INFERENCE_NITER,algorithm='swap')        
        

class CRF(object):
    """
    CRF postprocessing
    -uses dense CRF with Gaussian edge potential (https://arxiv.org/abs/1210.5644)
    -unary energy is derived from predicted probability map
    -pairwise energy is Potts function (parameters optimized using grid search)
    """

    def __init__(self, inference_niter=10):
        self.INFERENCE_NITER = inference_niter #number of mean field iteration in dense CRF
        self.w1 = 1.
        self.scale = 1000.
        
    def fit(self, X, y, idx):
        [self.nRows,self.nCols,self.nClasses] = X.shape
        self.nPixels = self.nRows*self.nCols
        prob_list = X.reshape([self.nPixels,self.nClasses])
        self.uEnergy = -np.log(prob_list.transpose().astype('float32')) #energy=-log(probability)
        
        params = [np.logspace(-3,3,10) for i in range(2)]      
        params = np.meshgrid(*params)
        params = np.vstack([x.ravel() for x in params]).T
        
        best_score = 0.0

        print('Fitting Fully-Connected CRF...')
        for i in range(params.shape[0]):
            crf = dcrf.DenseCRF2D(self.nRows,self.nCols,self.nClasses)
            crf.setUnaryEnergy(np.ascontiguousarray(self.uEnergy))
            crf.addPairwiseGaussian(sxy=params[i,0], compat=params[i,1])

            Q = crf.inference(self.INFERENCE_NITER)
            pred = np.argmax(Q,axis=0)
            score = np.sum(pred[idx]==y)/float(y.size)

            if score > best_score:
                self.w1 = params[i,1]
                self.scale = params[i,0]
                best_score = score

        params = [np.logspace(np.log10(x)-1,np.log10(x)+1,10) for x in [self.scale, self.w1]] 
        params = np.meshgrid(*params)
        params = np.vstack([x.ravel() for x in params]).T
        
        best_score = 0.0


        print('Finetuning Fully-Connected CRF...')
        for i in range(params.shape[0]):
            crf = dcrf.DenseCRF2D(self.nRows,self.nCols,self.nClasses)
            crf.setUnaryEnergy(np.ascontiguousarray(self.uEnergy))
            crf.addPairwiseGaussian(sxy=params[i,0], compat=params[i,1])

            Q = crf.inference(self.INFERENCE_NITER)
            pred = np.argmax(Q,axis=0)
            score = np.sum(pred[idx]==y)/float(y.size)
            
            if score > best_score:
                self.w1 = params[i,1]
                self.scale = params[i,0]
                best_score = score
        
        return 1         
        
    def predict(self, X):
        [self.nRows,self.nCols,self.nClasses] = X.shape
        self.nPixels = self.nRows*self.nCols
        prob_list = X.reshape([self.nPixels,self.nClasses])
        self.uEnergy = -np.log(prob_list.transpose().astype('float32'))
        
        crf = dcrf.DenseCRF2D(self.nRows,self.nCols,self.nClasses)
        crf.setUnaryEnergy(np.ascontiguousarray(self.uEnergy))
        crf.addPairwiseGaussian(sxy=self.scale, compat=self.w1)
       
        Q = crf.inference(self.INFERENCE_NITER)
        pred = np.argmax(Q,axis=0)
        return pred.reshape([self.nRows,self.nCols])

