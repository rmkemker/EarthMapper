"""
@author: ubg9540
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class RandomForestWorkflow(BaseEstimator, ClassifierMixin):
    
    """
    RandomForest (Classifier)
    
    Args:
        n_jobs (int, optional): Number of jobs to run in parallel for both fit 
            and predict. (Default: 8)
        verbosity (int, optional): Controls the verbosity of GridSearchCV: the 
            higher the number, the more messages. (Default: 10)
        refit (bool, optional): Whether to refit during the fine-tune search. 
            (Default: True)
        scikit_args (dict, optional)
    """
    
    def __init__(self, n_jobs=8, verbosity = 10, refit=True, scikit_args={}):
        self.n_jobs = n_jobs
        self.verbose = verbosity
        self.refit = refit
        self.scikit_args = scikit_args
                
    def fit(self, train_data, train_labels, val_data, val_labels):
        """
        Fits to training data.
        
        Args:
            train_data (ndarray): Training data.
            train_labels (ndarray): Training labels.
            val_data (ndarray): Validation data.
            val_labels (ndarray): Validation labels.
        """
        split = np.append(-np.ones(train_labels.shape, dtype=np.float32),
                  np.zeros(val_labels.shape, dtype=np.float32))
        ps = PredefinedSplit(split)

        sh = train_data.shape
        train_data = np.append(train_data, val_data , axis=0)
        train_labels = np.append(train_labels , val_labels, axis=0)
        del val_data, val_labels
        
        model = RandomForestClassifier(n_jobs=self.n_jobs,
                                       **self.scikit_args)        
    
        params = {'n_estimators':np.arange(1,1000,50)}    
        #Coarse search      
        gs = GridSearchCV(model, params, refit=False, n_jobs=self.n_jobs,  
                          verbose=self.verbose, cv=ps)
        gs.fit(train_data, train_labels)
        
        #Fine-Tune Search
        params = {'n_estimators':np.arange(gs.best_params_['n_estimators']-50,
                 gs.best_params_['n_estimators']+50)}    
        
        self.gs = GridSearchCV(model, params, refit=self.refit, n_jobs=self.n_jobs,  
                          verbose=self.verbose, cv=ps)
        self.gs.fit(train_data, train_labels)
        
        if not self.refit:
            model.set_params(n_estimators=gs.best_params_['n_estimators'])
            self.gs = model
            self.gs.fit(train_data[:sh[0]], train_labels[:sh[0]])         
        
#        return self.gs.fit(train_data, train_labels)
                
    def predict(self, test_data):
        """
        Performs classification on samples in test_data.
        
        Args:
            test_data (ndarray): Test data to be classified.
        
        Returns:
            ndarray of predicted class labels for test_data.
        """
        return self.gs.predict(test_data)
        
        
    def predict_proba(self, test_data):
        """
        Computes probabilities of possible outcomes for samples in test_data.
        
        Args:
            test_data (ndarray): Test data that will be used to compute 
                probabilities.
        
        Returns:
            Array of the probability of the sample for each class in the 
                model.
        """
        return self.gs.predict_proba(test_data)