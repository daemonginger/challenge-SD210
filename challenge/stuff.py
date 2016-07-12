%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import mode
from collections import defaultdict

train_fname = 'data/train.csv'
test_fname = 'data/test.csv'
df = pd.read_csv(train_fname, sep=';')
df_test = pd.read_csv(test_fname, sep=';')
y = df['VARIABLE_CIBLE'].values == 'GRANTED'
df = df.drop('VARIABLE_CIBLE', axis=1)
df = pd.concat((df, df_test)).reset_index()
df = df.drop('index', axis=1)

f_con = ['cited_age_min','cited_age_median','cited_age_max','cited_age_mean',
        'pct_NB_IPC','pct_NB_IPC_LY','oecd_NB_BACKWARD_PL','oecd_NB_BACKWARD_NPL',
        'IDX_ORIGIN','IDX_RADIC']
f_cat = ['APP_NB','APP_NB_PAYS','APP_NB_TYPE','VOIE_DEPOT','COUNTRY','SOURCE_BEGIN_MONTH','FISRT_APP_COUNTRY',
         'FISRT_APP_TYPE','LANGUAGE_OF_FILLING','FIRST_CLASSE','TECHNOLOGIE_SECTOR','TECHNOLOGIE_FIELD','MAIN_IPC',
         'FISRT_INV_COUNTRY','FISRT_INV_TYPE','SOURCE_CITED_AGE','SOURCE_IDX_ORI','SOURCE_IDX_RAD',
         'NB_CLASSES','NB_ROOT_CLASSES','NB_SECTORS','NB_FIELDS','INV_NB','INV_NB_PAYS','INV_NB_TYPE',
         'NB_BACKWARD_NPL','NB_BACKWARD_XY','NB_BACKWARD_I','NB_BACKWARD_AUTRE','NB_BACKWARD_PL',
         'NB_BACKWARD','oecd_NB_ROOT_CLASSES','PRIORITY_MONTH','FILING_MONTH','PUBLICATION_MONTH','BEGIN_MONTH','cited_n']

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
	  
