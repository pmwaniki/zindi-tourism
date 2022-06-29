import os


from pytorch_tabnet.tab_network import TabNet

import torch
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import log_loss

import numpy as np
import pandas as pd

from settings import data_dir, result_dir, log_dir

experiment="Blending - MLP"
device='cuda' if torch.cuda.is_available() else 'cpu'
if device=='cpu':
    resources={"cpu": 1, "gpu": 0}
else:
    resources = {"cpu": 2, "gpu": 0.2}
train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
test=pd.read_csv(os.path.join(data_dir,"Test.csv"))

#COUNTRY
tab_countries=train['country'].value_counts()
dominant_countries=tab_countries[tab_countries>50].index.values
train['country']=train['country'].map(lambda x: x if x in dominant_countries else "Others")
test['country']=test['country'].map(lambda x: x if x in dominant_countries else "Others")
labels=['High Cost', 'Higher Cost', 'Highest Cost', 'Low Cost',
       'Lower Cost', 'Normal Cost']
#PSEUDO Labels
pseudo_train_files = {
    'semi-supervised': 'Pseudo Labels - semi-supervised.csv',
    'mlp': 'Pseudo Labels - MLP.csv',
    'catboost': 'Pseudo labels -boosting.csv',
    # 'tabnet':'Pseudo Labels - Tabnet.csv',
}
psuedo_test_files = {
    'semi-supervised': 'MLP semi-supervised.csv',
    'mlp': 'MLP baseline.csv',
    'catboost': 'Catboost baseline.csv',
    # 'tabnet':'Tabnet baseline.csv'
}

pseudo_train=train[['Tour_ID','cost_category']].copy()
pseudo_test=test['Tour_ID'].copy()
for m in pseudo_train_files.keys():
    prob_train=pd.read_csv(os.path.join(result_dir,pseudo_train_files[m]))
    prob_test=pd.read_csv(os.path.join(result_dir,psuedo_test_files[m]))
    prob_test['Tour_ID']=prob_test['Test_ID']
    prob_train=prob_train[['Tour_ID']+labels]
    prob_test = prob_test[['Tour_ID'] + labels]

    prob_train.rename(columns={l:f'{l} - {m}' for l in labels},inplace=True)
    prob_test.rename(columns={l: f'{l} - {m}' for l in labels}, inplace=True)
    pseudo_train=pd.merge(prob_train,pseudo_train,on="Tour_ID")
    pseudo_test=pd.merge(prob_test,pseudo_test,on="Tour_ID")

predictors_extra=list(pseudo_test.columns[1:])

X_train=pseudo_train[predictors_extra].copy()
label_enc=LabelEncoder()
y_train=label_enc.fit_transform(pseudo_train['cost_category'])

X_test=pseudo_test[predictors_extra].copy()
ids_test=pseudo_test['Tour_ID']

grid={
    'clf__C':loguniform(0.0001,1000),
    'pca__n_components':[2,5,10,15,24]
}

model=Pipeline([
    # ('scl',StandardScaler()),
    ('pca',PCA(n_components=5)),
    ('clf',LogisticRegression(max_iter=10000))
])

model_cv=RandomizedSearchCV(model,param_distributions=grid,n_iter=500,
                            scoring='neg_log_loss',n_jobs=-1,
                            cv=5)
model_cv.fit(X_train,y_train)
print(model_cv.best_score_)

test_predictions=model_cv.predict_proba(X_test)
test_df=pd.DataFrame(test_predictions,columns=label_enc.classes_)
test_df['Test_ID']=ids_test

test_df.to_csv(os.path.join(result_dir,"Blending Logistic.csv"),index=False)