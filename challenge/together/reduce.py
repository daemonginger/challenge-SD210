import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import sys

print('Starting loading')

y_fname = '../data/y_train_1.csv'
df_y = pd.read_csv(y_fname, sep=';')
n_samples = df_y.shape[0]
y_train = df_y['VARIABLE_CIBLE'].values == 'GRANTED'

print('Finished loading')

cut = int(n_samples*3/4)
y_valid = y_train[cut:]

y_pred_valid = np.zeros(y_valid.shape[0])
for i in range(1,int(sys.argv[1])+1):
	y_pred_valid += np.loadtxt('../partial/'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(i))

print('Score sur la validation : {}'.format(roc_auc_score(y_valid, y_pred_valid/int(sys.argv[1]))))
print('Parametres : n_estimators = {0} ; n_leaf = {1}'.format(int(sys.argv[1])*int(sys.argv[2]),int(sys.argv[3])))

with open("scores", "a") as f:
	f.write('Score sur la validation : {}\n'.format(roc_auc_score(y_valid, y_pred_valid/int(sys.argv[1]))))
	f.write('Parametres : n_estimators = {0} ; n_leaf = {1}\n\n'.format(int(sys.argv[1])*int(sys.argv[2]),int(sys.argv[3])))


