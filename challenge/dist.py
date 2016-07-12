#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
import time
import sys

fname = 'data/train.csv'
df = pd.read_csv(fname, sep=';')
y_train = df['VARIABLE_CIBLE'].values == 'GRANTED'
n_samples = df.shape[0]

f_cat = ['VOIE_DEPOT','COUNTRY','SOURCE_BEGIN_MONTH','FISRT_APP_COUNTRY','FISRT_APP_TYPE','LANGUAGE_OF_FILLING',
         'TECHNOLOGIE_SECTOR','TECHNOLOGIE_FIELD','FISRT_INV_COUNTRY','FISRT_INV_TYPE','FIRST_CLASSE',
         'SOURCE_CITED_AGE','SOURCE_IDX_ORI','MAIN_IPC','SOURCE_IDX_RAD']


f_cat_2 = ['VOIE_DEPOT','COUNTRY','SOURCE_BEGIN_MONTH','FISRT_APP_COUNTRY','FISRT_APP_TYPE','LANGUAGE_OF_FILLING',
         'TECHNOLOGIE_SECTOR','TECHNOLOGIE_FIELD','FISRT_INV_COUNTRY','FISRT_INV_TYPE',
         'SOURCE_CITED_AGE','SOURCE_IDX_ORI','SOURCE_IDX_RAD']

X_train = df[f_cat].values

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
	  
encoder = MultiColumnLabelEncoder()
X_train = encoder.fit_transform(df[f_cat]).values

ohencoder = OneHotEncoder()
X_train_ohe = ohencoder.fit_transform(X_train)
print(X_train_ohe.shape)

def scorer(estimator, X, y):
    return roc_auc_score(y, estimator.predict_proba(X)[:,1])
n_taken = 1000
params = {'n_estimators':[20]}
clf = GridSearchCV(RandomForestClassifier(), params, scorer, n_jobs=2)
start = time.time()
clf.fit(X_train_ohe[0:n_taken], y_train[0:n_taken])
print("{} secondes pour fit !".format(time.time() - start))
print("Score honorable sur la CV : {}".format(clf.best_score_))
print("Meilleurs pamametres : {}".format(clf.best_params_))