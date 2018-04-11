"""
Name: svm_workflow.py
Author: Ronald Kemker
Description: Classification pipeline for support vector machine where train and
             validation folds are pre-defined.
Note:
Requires scikit-learn
"""


from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SVMWorkflow(BaseEstimator, ClassifierMixin):
    
    """
    Support Vector Machine (Classifier)
    
    Args:
        kernel(str, optional): Type of SVM kernel. Can be either 'linear' or 
            'rbf'. (Default: 'linear')
        n_jobs(int, optional): Number of jobs to run in parallel for 
            GridSearchCV. (Default: 8)
        verbosity(int ,optional): Controls the verbosity of GridSearchCV: the 
            higher the number, the more messages. (Default: 10)
        probability(bool, optional): Whether to enable probability estimates. 
            (Default: False)
        refit(bool, optional): For GridSearchCV. Whether to refit the best 
            estimator with the entire dataset. (Default: True)
        scikit_args(dict, optional)
    """
    
    def __init__(self, kernel='linear', n_jobs=8, verbosity = 0, 
                 probability=False, refit=True, scikit_args={}):
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.probability = probability
        self.scikit_args = scikit_args
        self.refit = refit
                
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
        
        if self.kernel == 'linear':
            if self.probability:
                clf = SVC(kernel='linear', class_weight='balanced',
                          random_state=6, decision_function_shape='ovr',
                          max_iter=1000, probability=self.probability,
                          **self.scikit_args)
            else:
                clf = LinearSVC(class_weight='balanced', dual=False,
                                random_state=6, multi_class='ovr',
                                max_iter=1000, **self.scikit_args)
        
            #Cross-validate over these parameters
            params = {'C': 2.0**np.arange(-9,16,2,dtype=np.float)}
        elif self.kernel == 'rbf':
            clf = SVC(random_state=6, class_weight='balanced', cache_size=16000,
                      decision_function_shape='ovr',max_iter=1000, tol=1e-4, 
                      probability=self.probability, **self.scikit_args)            
            params = {'C': 2.0**np.arange(-9,16,2,dtype=np.float),
                      'gamma': 2.0**np.arange(-15,4,2,dtype=np.float)}

        #Coarse search      
        gs = GridSearchCV(clf, params, refit=False, n_jobs=self.n_jobs,  
                          verbose=self.verbosity, cv=ps)
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
        
        self.gs = GridSearchCV(clf, params, refit=self.refit, n_jobs=self.n_jobs,  
                          verbose=self.verbosity, cv=ps)
        self.gs.fit(train_data, train_labels)
        
        if not self.refit:
            clf.set_params(C=gs.best_params_['C'])
            if self.kernel == 'rbf':
                clf.set_params(gamma=gs.best_params_['gamma'])
            self.gs = clf
            self.gs.fit(train_data[:sh[0]], train_labels[:sh[0]])
            
                
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

