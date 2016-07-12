import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import operator

train_fname = '../data/train.csv'
test_fname = '../data/test.csv'
df = pd.read_csv(train_fname, sep=';')
df_test = pd.read_csv(test_fname, sep=';')
(n_samples,n_variables) = df.shape

f_con = ['APP_NB','APP_NB_PAYS','APP_NB_TYPE','NB_CLASSES','NB_ROOT_CLASSES','NB_SECTORS','NB_FIELDS','INV_NB',
        'INV_NB_PAYS','INV_NB_TYPE','cited_n','cited_age_min','cited_age_median','cited_age_max','cited_age_mean',
        'cited_age_std','NB_BACKWARD_NPL','NB_BACKWARD_XY','NB_BACKWARD_I','NB_BACKWARD_AUTRE','NB_BACKWARD_PL',
        'NB_BACKWARD','pct_NB_IPC','pct_NB_IPC_LY','oecd_NB_ROOT_CLASSES','oecd_NB_BACKWARD_PL','oecd_NB_BACKWARD_NPL',
        'IDX_ORIGIN','IDX_RADIC','PRIORITY_MONTH','FILING_MONTH','PUBLICATION_MONTH','BEGIN_MONTH']
f_cat = ['VOIE_DEPOT','COUNTRY','SOURCE_BEGIN_MONTH','FISRT_APP_COUNTRY','FISRT_APP_TYPE','LANGUAGE_OF_FILLING',
        'FIRST_CLASSE','TECHNOLOGIE_SECTOR','TECHNOLOGIE_FIELD','MAIN_IPC','FISRT_INV_COUNTRY','FISRT_INV_TYPE','SOURCE_CITED_AGE',
        'SOURCE_IDX_ORI','SOURCE_IDX_RAD']

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
	  
X_train_cat = df[f_cat].values
X_test_cat = df_test[f_cat].values

cnt = defaultdict(lambda: 0)

for i in range(X_train_cat.shape[0]):
    cnt[X_train_cat[i][6]] += 1
for i in range(X_test_cat.shape[0]):
    cnt[X_test_cat[i][6]] += 1
stuff = dict(sorted(cnt.iteritems(), key=operator.itemgetter(1), reverse=True)[:10000])    

for i in range(X_train_cat.shape[0]):
    if(not X_train_cat[i][6] in stuff):
        X_train_cat[i][6] = '(MISSING)'
for i in range(X_test_cat.shape[0]):
    if(not X_test_cat[i][6] in stuff):
        X_test_cat[i][6] = '(MISSING)'
        
encoder = MultiColumnLabelEncoder()
X_cat = encoder.fit_transform(pd.DataFrame(np.concatenate([X_train_cat,X_test_cat], axis=0),columns=f_cat)).values

X_train_cat = X_cat[0:n_samples,:]
X_test_cat = X_cat[n_samples:,:]

X_train = np.concatenate((X_train_con,X_train_cat), axis=1)
X_test = np.concatenate((X_test_con,X_test_cat), axis=1)

pd.DataFrame(data=X_train, columns=f_con + f_cat).to_csv(path_or_buf='../data/train_6.csv', sep=';')
pd.DataFrame(data=X_test, columns=f_con + f_cat).to_csv(path_or_buf='../data/test_6.csv', sep=';')
