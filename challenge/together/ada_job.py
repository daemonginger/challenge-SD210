
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
import time
import sys

print('Starting loading')

train_fname = '../data/train_3.csv'
test_fname = '../data/test_3.csv'
df = pd.read_csv(train_fname, sep=';')
df_test = pd.read_csv(test_fname, sep=';')
y_fname = '../data/y_train_1.csv'
df_y = pd.read_csv(y_fname, sep=';')
(n_samples,n_variables) = df.shape

print('Finished loading')

f_con = ['APP_NB','APP_NB_PAYS','APP_NB_TYPE','NB_CLASSES','NB_ROOT_CLASSES','NB_SECTORS','NB_FIELDS','INV_NB',
        'INV_NB_PAYS','INV_NB_TYPE','cited_n','cited_nmiss','cited_age_min','cited_age_median','cited_age_max','cited_age_mean',
        'cited_age_std','NB_BACKWARD_NPL','NB_BACKWARD_XY','NB_BACKWARD_I','NB_BACKWARD_AUTRE','NB_BACKWARD_PL',
        'NB_BACKWARD','pct_NB_IPC','pct_NB_IPC_LY','oecd_NB_ROOT_CLASSES','oecd_NB_BACKWARD_PL','oecd_NB_BACKWARD_NPL',
        'IDX_ORIGIN','IDX_RADIC','PRIORITY_MONTH','FILING_MONTH','PUBLICATION_MONTH','BEGIN_MONTH']
f_cat = ['VOIE_DEPOT','COUNTRY','SOURCE_BEGIN_MONTH','FISRT_APP_COUNTRY','FISRT_APP_TYPE','LANGUAGE_OF_FILLING',
        'FIRST_CLASSE','TECHNOLOGIE_SECTOR','TECHNOLOGIE_FIELD','MAIN_IPC','FISRT_INV_COUNTRY','FISRT_INV_TYPE','SOURCE_CITED_AGE',
        'SOURCE_IDX_ORI','SOURCE_IDX_RAD']

X_train = df.values[:,1:]
X_test = df_test.values[:,1:]
y_train = df_y['VARIABLE_CIBLE'].values == 'GRANTED'

print('Starting encoding')

encoder = OneHotEncoder(categorical_features=range(len(f_con),len(f_con) + len(f_cat)))
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print('Finished encoding')
print('Starting training')

X_train_2 = csr_matrix(X_train)
y_train_2 = y_train
cut = int(n_samples*3/4)
X_train, X_valid, y_train, y_valid = X_train_2[:cut], X_train_2[cut:], y_train_2[:cut], y_train_2[cut:]

n_thread = -1
n_tree = int(sys.argv[2])
n_leaf = int(sys.argv[3])
rate = float(sys.argv[4])
clf = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=n_leaf, min_samples_leaf=1, min_samples_split=1), n_estimators=n_tree, learning_rate=rate)
clf.fit(X_train,y_train)
y_pred_valid = clf.predict_proba(X_valid)[:, 1]
print(y_pred_valid)
print(y_valid)
print("Score sur la validation : {}".format(roc_auc_score(y_valid, y_pred_valid)))
np.savetxt('../partial/'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+sys.argv[1]+'_ada',y_pred_valid)

# 0.706617810855 'n_estimators': 500 'max_leaf_nodes': 10000. YOLOOOOOOOO
# 0.7084 - 0.7088 'n_estimators': 2000 - 20000 'max_leaf_nodes': 20000. :)
