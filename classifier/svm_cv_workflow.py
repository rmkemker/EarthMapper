"""
Name: svm_cv_workflow.py
Author: Ronald Kemker
Description: Classification pipeline for support vector machine which uses 
             k-fold cross-validation
Note:
Requires scikit-learn
"""

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class SVM_Workflow():
    
    def __init__(self, kernel='linear', n_jobs=8, probability=False,
                 verbosity = 10,  cv=3, max_iter=1000, scikit_args={}):
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.probability = probability
        self.verbose = verbosity
        self.cv = cv
        self.max_iter = max_iter
        self.scikit_args = scikit_args

        
    def fit(self, train_data, train_labels, X_val=None, y_val=None):
                
        if self.kernel == 'linear':
            if self.probability:
                clf = SVC(kernel='linear', class_weight='balanced',
                          random_state=6, multi_class='ovr',
                          max_iter=1000, probability=self.probability,
                          **self.scikit_args)
            else:
                clf = LinearSVC(class_weight='balanced', dual=False,
                                random_state=6, multi_class='ovr',
                                max_iter=1000, **self.scikit_args)
            params = {'C': 2.0**np.arange(-9,16,2,dtype=np.float)}

        elif self.kernel == 'rbf':
            clf = SVC(random_state=6, class_weight='balanced', cache_size=8000,
                      decision_function_shape='ovr',max_iter=self.max_iter, 
                      tol=1e-4)            
            params = {'C': 2.0**np.arange(-9,16,2,dtype=np.float),
                      'gamma': 2.0**np.arange(-15,4,2,dtype=np.float)}

        #Coarse search      
        gs = GridSearchCV(clf, params, refit=False, n_jobs=self.n_jobs,  
                          verbose=self.verbose, cv=self.cv)
        gs.fit(train_data, train_labels)
        
        #Fine-Tune Search
        if self.kernel == 'linear':
            best_C = np.log2(gs.best_params_['C'])
            params = {'C': 2.0**np.linspace(best_C-2,best_C+2,10,
                                            dtype=np.float)}
        elif self.kernel == 'rbf':
            best_C = np.log2(gs.best_params_['C'])
            best_G = np.log2(gs.best_params_['gamma'])
            params = {'C': 2.0**np.linspace(best_C-2,best_C+2,10,
                                            dtype=np.float),
                      'gamma': 2.0**np.linspace(best_G-2,best_G+2,10,
                                                dtype=np.float)}            
        
        self.gs = GridSearchCV(clf, params, refit=True, n_jobs=self.n_jobs,  
                          verbose=self.verbose, cv=self.cv)
        
        self.gs.fit(train_data, train_labels)
        
        return 1
    
                
    def predict(self, test_data):
        return self.gs.predict(test_data)
        
    def predict_proba(self, test_data):
        return self.gs.predict_proba(test_data)        
