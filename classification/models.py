import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.ensemble import VotingRegressor, VotingClassifier
import sklearn.metrics as metric_lib
from statistics import mean
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .common_functions import get_path
import importlib

class model_search:
    
    """
    Pipeline that uses grid-search, to find best models for a certain dataframe.
    """
    
    def __init__(self, X=None, y=None, params=None, train=True, break_on_error=False, cv=10, n_jobs=-1, files=None, csv='report_model.csv', folder='sk_models'):
        
        self.X = X
        self.y = y
        self.params = params
        self.break_on_error = break_on_error
        self.cv = cv
        self.n_jobs = n_jobs
        self.files = files
        self.csv = csv
        
        self.untrained_models = []
        self.path = get_path(folder, opt_print=False)
        self.df = None
        self.train = False
        
        if train is True:
            if X and y and params:
                
                self.train = True
                self.fit()
            
            else:
                raise AttributeError('Define data and parameters to fit')
                
        else:
            self.load()
        
        
    def fit(self, _return=False):
        
        if self.train is True:
            
            report = []
            cols = ['Model', 'Type', 'Score', 'OneVSRest', 'Parameters', 'File']

            for i, param in enumerate(tqdm(self.params, desc="Classifiers")):

                if self.files:
                    file = self.files[i]
                else:
                    file = '{}.pkl'.format(param['model'])

                try:
                    module = importlib.import_module("sklearn.{}".format(param['module']))
                    model = getattr(module, param['model'])
                except:
                    if self.break_on_error is True:
                        raise ValueError('Model {} does not exist on package {}'.format(param['model'], param['module']))
                    else:
                        self.untrained_models.append([param['model'], 'package'])
                        continue

                try:
                    model = model(random_state=42)
                except:
                    model = model()

                try:
                    # Grid Search
                    grid_search = GridSearchCV(model, param["param_grid"], n_jobs=self.n_jobs, cv=self.cv)
                    grid_search.fit(self.X[model._estimator_type], self.y[model._estimator_type])
                    # Best Model
                    model = grid_search.best_estimator_
                    hyper_param = model.get_params()
                    score = grid_search.best_score_
                    # OnevsRest
                    ovr_score = None
                    if model._estimator_type == 'classifier':
                        ovr_model = OneVsRestClassifier(model)
                        ovr_score = mean(cross_val_score(ovr_model, self.X['classifier'], self.y['classifier'], cv=self.cv, n_jobs=self.n_jobs))
                        ovr_model.fit(self.X['classifier'], self.y['classifier'])
                        joblib.dump(ovr_model, os.path.join(self.path, 'ovr_{}'.format(file)))
                    
                    joblib.dump(model, os.path.join(self.path, file))
                    report.append([param['model'], model._estimator_type, score, ovr_score, hyper_param, file])

                except:
                    if self.break_on_error is True:
                        raise ValueError('Model {} had fitting error'.format(param['model']))
                    else:
                        self.untrained_models.append([param['model'], 'fitting'])
                        continue
            
            self.untrained_models= pd.DataFrame(self.untrained_models, columns=['Model', 'Error'])
            
            if len(self.untrained_models) > 0:
                print('{} Untrained Model'.format(len(self.untrained_models)))
            else:
                print('All Models Trained')
            
            df = pd.DataFrame(report, columns=cols)
            self.df = df
            df.to_csv(self.csv)
            if _return is True:
                return df
        
        
    def load(self, _return=False):
        
        if not self.df:
            try:
                self.df = pd.read_csv(self.csv, index_col=0)
            except:
                raise AttributeError('Untrained Models- Fit models before loading')
        
        if _return is True:
                self.df
                
class analyse_model:
    
    """
    Pipeline fitting model and generating advanced metrics.
    """
    
    def __init__(self, params, ovr=False, class_names=None, split=0.3, random_state=42, plot=True):
        
        #Load Model
        
        if type(params) is dict:
            params = [params]
        models = []
        
        for param in params:
            try:
                module = importlib.import_module("sklearn.{}".format(param['module']))
                model = getattr(module, param['model'])
            except:
                raise ValueError('Model {} does not exist on package {}'.format(param['model'], param['module']))

            try:
                try:
                    model = model(**param['hp'])
                except TypeError:
                    model = model(**eval(param['hp']))

                if ovr is True:
                    model = OneVsRestClassifier(model)
            except:
                raise AttributeError('No model parameters defined')
                
            models.append((param['model'], model))
            
        if len(models) == 1:
            model = models[0][1]
            
        else:
            if models[0][1]._estimator_type == 'classifier':
                model = VotingClassifier(estimators=models, voting='hard')
            elif models[0][1]._estimator_type == 'regressor':
                model = VotingRegressor(estimators=models)
            else:
                model = None
        
        
        # Define vars
        
        self.model = model
        self.split = split
        self.random_state = random_state
        self.plot = plot
        self.class_names = class_names
        self.fitted = False
        
    def fit(self, X, y):
        if self.fitted is False:  
            try:
                self.model.fit(X, y)
                self.fitted = True
            except:
                raise ValueError('Fitting Error')
            
    def test(self, X, y):
        
        if self.fitted is True:
            
            metrics = None
            
            if self.model._estimator_type == 'classifier':

                np.set_printoptions(precision=2)

                # Metrics
                y_pred = self.model.predict(X)
                metrics = classification_report(y, y_pred)

                if self.plot is True:
                    # Plot
                    plt.figure(figsize=(18, 8));
                    disp = plot_confusion_matrix(self.model, X, y,
                                                 display_labels=self.class_names,
                                                 cmap=plt.cm.Blues,
                                                 normalize='true')
                    disp.ax_.set_title("Normalized confusion matrix")
                    plt.show()
            
            elif self.model._estimator_type == 'regressor':
                metrics = []
                
                y_pred = self.model.predict(X)
                
                scoring_metrics = ['mean_absolute_error', 'explained_variance_score', 'mean_squared_error', 
                 'mean_squared_log_error', 'median_absolute_error', 'r2_score', 
                 'mean_poisson_deviance', 'mean_gamma_deviance', 'mean_tweedie_deviance']
                
                for metric in scoring_metrics:
                    
                    try:
                    
                        score_function = getattr(metric_lib, metric)
                        score = score_function(y, y_pred)
                        metrics.append([metric, score])
                    
                    except:
                        pass
                    
                metrics = pd.DataFrame(metrics, columns=['Metric', 'Score'])
                metrics.sort_values(['Metric'], inplace=True)
                metrics.reset_index(drop=True, inplace=True)
                

            return metrics
            
        else:
            raise AttributeError('Untrained Model- Fit models before testing')
        
    def fit_test(self, X, y):
        
        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split, random_state=self.random_state)
        
        self.fit(X_train, y_train)
        return self.test(X_test, y_test)
        
        
    
        
        
            
        