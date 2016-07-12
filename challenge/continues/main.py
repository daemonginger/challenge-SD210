#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

train_fname = '../data/train.csv'
test_fname = '../data/test.csv'
df = pd.read_csv(train_fname, sep=';')
df_test = pd.read_csv(test_fname, sep=';')
f_con = ['APP_NB','APP_NB_PAYS','APP_NB_TYPE','NB_CLASSES','NB_ROOT_CLASSES','NB_SECTORS','NB_FIELDS','INV_NB',
        'INV_NB_PAYS','INV_NB_TYPE','cited_n','cited_age_min','cited_age_median','cited_age_max','cited_age_mean',
        'cited_age_std','NB_BACKWARD_NPL','NB_BACKWARD_XY','NB_BACKWARD_I','NB_BACKWARD_AUTRE','NB_BACKWARD_PL',
        'NB_BACKWARD','pct_NB_IPC','pct_NB_IPC_LY','oecd_NB_ROOT_CLASSES','oecd_NB_BACKWARD_PL','oecd_NB_BACKWARD_NPL',
        'IDX_ORIGIN','IDX_RADIC']
(n_samples,n_variables) = (df.shape[0],len(f_con))

df = df.reindex(np.random.permutation(df.index));
X_train1 = df[f_con].values
y_train = df.VARIABLE_CIBLE == 'GRANTED'
X_test1 = df_test[f_con].values
imputer = Imputer()
# Imputer permet de combler les trous quand des donnees manquent. Par defaut il prend la moyenne de la derniere donnee vue
# et de la prochaine. Ces donnees ne sont donc pas normalisees.
X_train1 = imputer.fit_transform(X_train1)
X_test1 = imputer.fit_transform(X_test1)
# Normalisation des features.
scale(X_train1,copy=False);
scale(X_test1,copy=False);

X = np.concatenate((X_train1,X_test1), axis=0)
scale(X, copy=False);
pca = PCA(n_components=n_variables)
X = pca.fit_transform(X)
X_train = X[0:X_train1.shape[0]]
X_test = X[X_train1.shape[0]:]

n_taken = n_samples
model = make_pipeline(PolynomialFeatures(degree=2, include_bias = False), LogisticRegression(C=0.014))
model.fit(X_train1[0:n_taken], y_train[0:n_taken]);
y_pred_train = model.predict_proba(X_train1)[:, 1]
print('Score (optimiste) sur le train : %s' % roc_auc_score(y_train, y_pred_train))
print(model.score(X_train1[0:n_taken], y_train[0:n_taken]));

# Application au test set et sauvegarde de la soumission correspondante.
y_pred = model.predict_proba(X_test1)[:, 1]
np.savetxt('../subs/continue_polylogreg3.txt', y_pred, fmt='%s')

# Les scores sur le site sont <0.5... Sur-apprentissage ?