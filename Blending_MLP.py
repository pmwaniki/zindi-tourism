"""
This script is similar to mlp.py but the predictions from mlp.py, boosting.py and semi-supervised.py
 are concatenated to the imputs.

"""
import os
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from torch.optim import Adam

import ray
# ray.init("auto")
from modules import Classifier
from utils import permute_augmentation

ray.init( dashboard_host="0.0.0.0")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import log_loss

import numpy as np
import pandas as pd

from settings import data_dir, result_dir, log_dir

experiment="Blending-MLP"
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

pseudo_train=train[['Tour_ID']].copy()
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

predictors_extra=list(pseudo_train.columns[1:])
train_extended=pd.merge(train,pseudo_train,on="Tour_ID")
test_extended=pd.merge(test,pseudo_test,on="Tour_ID")

predictors=['country', 'age_group', 'travel_with', 'total_female',
       'total_male', 'purpose', 'main_activity', 'info_source',
       'tour_arrangement', 'package_transport_int', 'package_accomodation',
       'package_food', 'package_transport_tz', 'package_sightseeing',
       'package_guided_tour', 'package_insurance', 'night_mainland',
       'night_zanzibar', 'first_trip_tz']

cat_predictors=['country', 'age_group', 'travel_with',  'purpose', 'main_activity', 'info_source',
       'tour_arrangement', 'package_transport_int', 'package_accomodation',
       'package_food', 'package_transport_tz', 'package_sightseeing',
       'package_guided_tour', 'package_insurance',  'first_trip_tz']


X=train_extended[predictors+predictors_extra].copy()
label_enc=LabelEncoder()
y=label_enc.fit_transform(train_extended['cost_category'])

X_test=test_extended[predictors+predictors_extra].copy()
ids_test=test['Tour_ID']

cat_idxs=[]
cat_dims=[]
for i,col in enumerate(X.columns):
    if col in cat_predictors:
        if col == 'country':
            mode = "Others"
        else:
            mode = X[col].value_counts().index[0]
        X[col].fillna(mode,inplace=True)
        X_test[col].fillna(mode,inplace=True)
        enc=LabelEncoder()
        enc.fit(np.concatenate([X[col],X_test[col]]))
        X[col]=enc.transform(X[col])
        X_test[col] = enc.transform(X_test[col])
        if len(enc.classes_)>2:
            cat_idxs.append(i)
            cat_dims.append(len(enc.classes_))
        else:
            scl=StandardScaler()
            X[col] = scl.fit_transform(X[col].values.reshape(-1, 1))
            X_test[col] = scl.transform(X_test[col].values.reshape(-1, 1))

    else:
        median=X[col].median()
        X[col].fillna(median,inplace=True)
        X_test[col].fillna(median,inplace=True)
        scl = RobustScaler()
        X[col] = scl.fit_transform(X[col].values.reshape(-1, 1))
        X_test[col] = scl.transform(X_test[col].values.reshape(-1, 1))



configs={
    'lr':tune.loguniform(0.0001,0.1),
    'l2': tune.loguniform(0.00001,0.01),
    'batch_size' : tune.choice([32,64,128,256,512,1024]),
    # 'lambda_sparsity' : tune.loguniform(0.0000001,0.001),
    'emb_size':tune.choice([2,4,8,]),
    'dim_hidden':tune.choice([128,256,512]),
    'n_hidden':tune.choice([2,3,5,7,]),
    'dropout':tune.uniform(0.01,0.2),

    'p_corrupt': tune.uniform(0.05,0.2),
}
config={i:v.sample() for i,v in configs.items()}

def get_model(config):
    if device == 'cpu':
        torch.manual_seed(123)
    else:
        torch.cuda.manual_seed(123)
    model = Classifier(dim_x=X.shape[1], cat_idx=cat_idxs, cat_dims=cat_dims,
                       emb_size=config['emb_size'],n_hidden=config['n_hidden'],
                       dim_hidden=config['dim_hidden'],dropout=config['dropout'])

    return model

def get_optimizer(config,model):
    optimizer=torch.optim.Adam(params=model.parameters(),lr=config['lr'],weight_decay=config['l2'])
    return optimizer

def get_train_loader(config,X_train,y_train):
    dataset=TensorDataset(torch.tensor(X_train),torch.tensor(y_train,dtype=torch.float))
    train_loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,)
    # train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, sampler=None)
    return train_loader




def get_val_loader(X_valid,y_valid=None,batch_size=256):
    if y_valid is not None:
        dataset = TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid,dtype=torch.float))
    else:
        dataset = TensorDataset(torch.tensor(X_valid), )
    val_loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return val_loader




def train_fun(model,criterion,optimizer,train_loader,val_loader,
              device='cpu',scheduler=None,p_corrupt=0.0):
    model.train()
    train_loss = 0
    total_train_loss=0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device,dtype=torch.float), batch_y.to(device,dtype=torch.int64)
        with torch.no_grad():
            batch_x=permute_augmentation(batch_x,p_corrupt=p_corrupt)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss=loss+0
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader)
        total_train_loss += total_loss.item() / len(train_loader)

    model.eval()
    val_loss = 0
    total_val_loss=0
    val_pred = []
    val_obs = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device,dtype=torch.float), batch_y.to(device,dtype=torch.int64)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss = loss +0
            val_loss += loss.item() / len(val_loader)
            total_val_loss += total_loss.item()/len(val_loader)
            val_pred.append(logits.softmax(dim=1).cpu().numpy())
            val_obs.append(batch_y.cpu().numpy())

    if scheduler: scheduler.step(val_loss)
    val_pred = np.concatenate(val_pred)
    val_obs = np.concatenate(val_obs)
    logloss = log_loss(val_obs, val_pred,labels=[0,1,2,3,4,5])

    return train_loss,total_train_loss, val_loss,total_val_loss, logloss

X_test = X_test.values
test_loader = get_val_loader(X_test)
folds=KFold(n_splits=5,random_state=698,shuffle=True)
kfold_results=[]
fold_pred=[]
# pseudo_labels=[]
# pseudo_ids=[]
for i,(id_train,id_valid) in enumerate(folds.split(X,y)):
    X_train, X_valid=X.iloc[id_train,:],X.iloc[id_valid,:]
    y_train,y_valid=y[id_train],y[id_valid]
    # pseudo_ids.append(train['Tour_ID'].iloc[id_valid].values)

    X_train = X_train.values
    X_valid = X_valid.values

    class Trainer(tune.Trainable):
        def setup(self, config):
            self.model=get_model(config).to(device)
            self.optimizer=get_optimizer(config,self.model)
            # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=50,gamma=0.5)
            self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,
                                                                      patience=10,min_lr=1e-5)
            self.criterion=torch.nn.CrossEntropyLoss().to(device)
            self.train_loader=get_train_loader(config,X_train,y_train)
            self.val_loader=get_val_loader(X_valid,y_valid)
            self.p_corrupt=config['p_corrupt']


        def step(self):
            train_loss,total_train_loss,val_loss,total_val_loss,logloss=train_fun(self.model,self.criterion,
                                                                              self.optimizer,train_loader=self.train_loader,
                                              val_loader=self.val_loader,device=device,
                                              scheduler=self.scheduler,p_corrupt=self.p_corrupt)
            return {'train_loss':train_loss,'loss':val_loss,'logloss':logloss,
                    'total_train_loss':total_train_loss,'total_val_loss':total_val_loss}
        def save_checkpoint(self, checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save((self.model.state_dict(), self.optimizer.state_dict()), checkpoint_path)
            return checkpoint_path
        def load_checkpoint(self, checkpoint_path):
            model_state,optimizer_state=torch.load(checkpoint_path)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

    epochs=150
    scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=epochs,
            grace_period=50,
            reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "train-loss", "logloss", "training_iteration"])


    result = tune.run(
        Trainer,
        checkpoint_at_end=True,
        resources_per_trial=resources,
        config=configs,
        local_dir=os.path.join(log_dir,experiment),
        num_samples=100,
        name=f'fold{i}',
        resume=False,
        scheduler=scheduler,
        # stop=early_stopper,
        progress_reporter=reporter,
        reuse_actors=False,
        raise_on_failed_trial=False)

    metric='logloss';mode='min';scope='last-5-avg'
    best_trial = result.get_best_trial(metric, mode, scope=scope)
    best_result = best_trial.last_result
    print(best_result)
    kfold_results.append(best_result)

    best_config = result.get_best_config(metric, mode, scope=scope)
    best_checkpoint = result.get_best_checkpoint(best_trial, metric=metric, mode=mode)
    best_model_state, _ = torch.load(best_checkpoint)
    best_model = get_model(best_config).to(device)


    best_model.load_state_dict(best_model_state)
    best_model.eval()
    test_pred = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x[0].to(device, dtype=torch.float)
            logits = best_model(batch_x)
            test_pred.append(logits.softmax(dim=1).cpu().numpy())
    test_pred = np.concatenate(test_pred)
    fold_pred.append(test_pred)

    # valid_loader=get_val_loader(X_valid)
    # valid_pred = []
    # with torch.no_grad():
    #     for batch_x in valid_loader:
    #         batch_x = batch_x[0].to(device, dtype=torch.float)
    #         logits = best_model(batch_x)
    #         valid_pred.append(logits.softmax(dim=1).cpu().numpy())
    # valid_pred = np.concatenate(valid_pred)
    # pseudo_labels.append(valid_pred)

# pseudo_labels_df=pd.DataFrame(np.concatenate(pseudo_labels,axis=0),columns=label_enc.classes_)
# pseudo_labels_df['Tour_ID']=np.concatenate(pseudo_ids,axis=0)
test_predictions=np.stack(fold_pred).mean(axis=0)


test_df=pd.DataFrame(test_predictions)
test_df['Test_ID']=ids_test
test_df_long=pd.melt(test_df,id_vars='Test_ID',var_name='category',value_name='prob')
test_df_long['category']=label_enc.inverse_transform(test_df_long['category'].astype('int'))
test_df_wide=pd.pivot_table(test_df_long,values='prob',index='Test_ID',columns='category')

test_df_wide.to_csv(os.path.join(result_dir,f"{experiment}.csv"))
# pseudo_labels_df.to_csv(os.path.join(result_dir,"Pseudo Labels - MLP.csv"))

loglosses=[f['logloss'] for f in kfold_results]
print("losses:", loglosses)
print("Mean loss: ",np.mean(loglosses))
