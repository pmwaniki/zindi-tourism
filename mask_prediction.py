import os


from pytorch_tabnet.tab_network import TabNet

import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from torch.optim import Adam
import torch.nn.functional as F

import ray
# ray.init("auto")
from modules import ClassifierRegularized, MaskPrediction
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

experiment="Mask-Prediction"
device='cuda' if torch.cuda.is_available() else 'cpu'
if device=='cpu':
    resources={"cpu": 1, "gpu": 0}
else:
    resources = {"cpu": 2, "gpu": 0.2}
if device == 'cpu':
    torch.manual_seed(123)
else:
    torch.cuda.manual_seed(123)

train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
test=pd.read_csv(os.path.join(data_dir,"Test.csv"))

#COUNTRY
tab_countries=train['country'].value_counts()
dominant_countries=tab_countries[tab_countries>50].index.values
train['country']=train['country'].map(lambda x: x if x in dominant_countries else "Others")
test['country']=test['country'].map(lambda x: x if x in dominant_countries else "Others")



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


X=train[predictors].copy()
label_enc=LabelEncoder()
y=label_enc.fit_transform(train['cost_category'])

X_test=test[predictors].copy()
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

X_all=pd.concat([X,X_test],axis=0)

configs={
    'lr':tune.loguniform(0.0001,0.1),
    'lr_mask':tune.loguniform(0.00001,0.0001),
    # 'lr_outcome':tune.loguniform(0.0001,0.1),
    'l2': tune.loguniform(0.00001,0.01),
    'l2_mask': tune.loguniform(0.00001,0.01),
    'batch_size' : tune.choice([128,256,512,1024]),
    'lambda_l1' : tune.uniform(0.05,0.1),
    'emb_size':tune.choice([2,4,8,]),
    'dim_hidden':tune.choice([64,32,16]),
    'dim_z':tune.choice([32,16,64]),
    'n_hidden':tune.choice([1,3,]),
    'n_hidden_decoder':tune.choice([1,2]),
    'dropout':tune.uniform(0.01,0.2),
    'sd_noise_z':tune.loguniform(0.001,10.0)

    # 'p_corrupt': tune.uniform(0.05,0.3),
}
config={i:v.sample() for i,v in configs.items()}

def get_model(config):
    model = MaskPrediction(dim_x=X.shape[1], cat_idx=cat_idxs, cat_dims=cat_dims,
                       emb_size=config['emb_size'],n_hidden=config['n_hidden'],dim_z=config['dim_z'],
                       dim_hidden=config['dim_hidden'],dropout=config['dropout'])

    return model

def get_optimizer(config,model):
    optimizer=torch.optim.Adam(params=[
        {'params':model.encoder.parameters()},
        {'params':model.decoder.parameters(),'lr':config['lr_mask'],'weight_decay':config['l2_mask']},
    ],lr=config['lr'],weight_decay=config['l2'])
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

def get_unlabelled_loader(config,X,shuffle=False,drop_last=False):
    dataset=TensorDataset(torch.tensor(X))
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=shuffle,drop_last=drop_last)
    return loader

def discriminative_score(X,y):
    clf = Pipeline([
        ('threashold', VarianceThreshold()),
        ('scl', StandardScaler()),
        # ('clf',LogisticRegression(max_iter=5000)),
        ('clf', GaussianNB()),
    ])
    scores=cross_validate(clf,X,y,cv=KFold(10),n_jobs=1,scoring='neg_log_loss')['test_score']

    if np.all(np.isnan(scores)):
        print(f"!!!!!!!Collapsed encoder. mean={X.mean()},min={X.min()},max={X.max()} ")
        return np.nan
    return np.nanmean(scores)

def kl_loss(rho,embeddings,apply_sigmoid=True):
    if apply_sigmoid:
        embeddings=torch.sigmoid(embeddings)
    rho_hat = torch.mean(embeddings, 1)
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

labelled_loader=get_val_loader(X.values,y,batch_size=256)
def train_fun(model,optimizer,train_loader,val_loader,
              device='cpu',scheduler=None,
              lambda_l1=0.0,sd_noise_z=0.0
              ):

    model.train()
    # loss_weight = F.softmax(torch.randn(2,device=device), dim=-1, )
    train_loss = 0
    total_train_loss=0
    train_mask_loss=0
    for batch_x in train_loader:
        p_corrupt = np.random.uniform(0.05, 0.2)
        batch_x= batch_x[0].to(device,dtype=torch.float)
        with torch.no_grad():
            mask = torch.full_like(batch_x, p_corrupt, device=batch_x.device, dtype=batch_x.dtype)
            mask = mask.bernoulli()
            x_corrupt = torch.stack([batch_x[torch.randperm(batch_x.shape[0]), i]
                                     for i in range(batch_x.shape[1])], dim=1).to(batch_x.device)
            batch_x = (mask * x_corrupt) + ((1 - mask) * batch_x)
        z,logits_mask = model(batch_x)
        var_z=torch.abs(1-torch.var(z,dim=0).mean())
        # noise_z=torch.randn_like(z,device=z.device,dtype=z.dtype) * sd_noise_z
        # z=z + noise_z
        # loss_l1 = torch.norm(z,p=2,dim=1).mean()
        loss_l1=kl_loss(lambda_l1,z,apply_sigmoid=True)
        loss_mask=F.binary_cross_entropy_with_logits(logits_mask,mask)
        total_loss = loss_l1 + loss_mask+var_z
        # total_loss= torch.sum(torch.stack([loss,loss_mask]) * loss_weight)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += loss_l1.item() / len(train_loader)
        total_train_loss += total_loss.item() / len(train_loader)
        train_mask_loss += loss_mask.item()/len(train_loader)

    model.eval()
    val_loss = 0
    total_val_loss=0
    val_mask_loss=0

    with torch.no_grad():
        for batch_x in val_loader:
            p_corrupt = np.random.uniform(0.05, 0.2)
            batch_x= batch_x[0].to(device,dtype=torch.float)
            with torch.no_grad():
                mask = torch.full_like(batch_x, p_corrupt, device=batch_x.device, dtype=batch_x.dtype)
                mask = mask.bernoulli()
                x_corrupt = torch.stack([batch_x[torch.randperm(batch_x.shape[0]), i]
                                         for i in range(batch_x.shape[1])], dim=1).to(batch_x.device)
                batch_x = (mask * x_corrupt) + ((1 - mask) * batch_x)
            z, logits_mask = model(batch_x)
            var_z=torch.abs(1-torch.var(z,dim=0).mean())
            # loss_l1 = torch.norm(z,p=2,dim=1).mean()
            loss_l1 = kl_loss(lambda_l1, z, apply_sigmoid=True)
            loss_mask = F.binary_cross_entropy_with_logits(logits_mask, mask)
            total_loss = loss_l1 + loss_mask+var_z
            val_loss += loss_l1.item() / len(val_loader)
            total_val_loss += total_loss.item()/len(val_loader)
            val_mask_loss += loss_mask.item()/len(val_loader)

    labelled_z=[]
    val_obs=[]
    with torch.no_grad():
        for batch_x, batch_y in labelled_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device, dtype=torch.int64)
            mask = torch.full_like(batch_x, p_corrupt, device=batch_x.device, dtype=batch_x.dtype)
            mask = mask.bernoulli()
            x_corrupt = torch.stack([batch_x[torch.randperm(batch_x.shape[0]), i]
                                     for i in range(batch_x.shape[1])], dim=1).to(batch_x.device)
            batch_x = (mask * x_corrupt) + ((1 - mask) * batch_x)
            z = model.encoder(batch_x)
            labelled_z.append(z.cpu().numpy())
            val_obs.append(batch_y.cpu().numpy())

    labelled_z = np.concatenate(labelled_z)
    val_obs = np.concatenate(val_obs)
    logloss = discriminative_score(labelled_z, val_obs) * -1.0

    if scheduler: scheduler.step(val_mask_loss)


    return train_loss,total_train_loss, val_loss,total_val_loss, logloss,train_mask_loss,val_mask_loss

X_test = X_test.values
test_loader = get_val_loader(X_test)
folds=KFold(n_splits=5,random_state=548,shuffle=True)
kfold_results=[]
fold_pred=[]
# pseudo_labels=[]
# pseudo_ids=[]
for i,(id_train,id_valid) in enumerate(folds.split(X_all)):
    X_train, X_valid=X_all.iloc[id_train,:],X_all.iloc[id_valid,:]
    # y_train,y_valid=y[id_train],y[id_valid]
    # pseudo_ids.append(train['Tour_ID'].iloc[id_valid].values)

    X_train = X_train.values
    X_valid = X_valid.values

    class Trainer(tune.Trainable):
        def setup(self, config):
            self.model=get_model(config).to(device)
            self.optimizer=get_optimizer(config,self.model)
            # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=50,gamma=0.5)
            self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,
                                                                      patience=10,min_lr=1e-6)
            self.train_loader=get_unlabelled_loader(config,X_train,shuffle=True,drop_last=True)
            self.val_loader=get_unlabelled_loader(config,X_valid,shuffle=True,drop_last=True)
            self.lambda_l1=config['lambda_l1']
            self.sd_noise_z=config['sd_noise_z']


        def step(self):
            train_loss, total_train_loss, val_loss, total_val_loss, logloss, \
            train_mask_loss, val_mask_loss = train_fun(self.model,
                                                       self.optimizer, train_loader=self.train_loader,
                                                       val_loader=self.val_loader, device=device,
                                                       scheduler=self.scheduler,
                                                       lambda_l1=self.lambda_l1,
                                                       sd_noise_z=self.sd_noise_z
                                                       )
            return {'train_loss':train_loss,'loss':val_loss,'logloss':logloss,
                    'total_train_loss':total_train_loss,'total_val_loss':total_val_loss,
                    'train_mask_loss':train_mask_loss,'val_mask_loss':val_mask_loss}
        def save_checkpoint(self, checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save((self.model.state_dict(), self.optimizer.state_dict()), checkpoint_path)
            return checkpoint_path
        def load_checkpoint(self, checkpoint_path):
            model_state,optimizer_state=torch.load(checkpoint_path)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

    epochs=500
    scheduler = ASHAScheduler(
            metric="logloss",
            mode="min",
            max_t=epochs,
            grace_period=100,
            reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "train-loss", "logloss", "training_iteration"])


    # early_stopper=EarlyStopper(metric='loss',tolerance=10,mode='min')
    result = tune.run(
        Trainer,
        checkpoint_at_end=True,
        resources_per_trial=resources,
        config=configs,
        local_dir=os.path.join(log_dir,experiment),
        num_samples=300,
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
    break

#     best_config = result.get_best_config(metric, mode, scope=scope)
#     best_checkpoint = result.get_best_checkpoint(best_trial, metric=metric, mode=mode)
#     best_model_state, _ = torch.load(best_checkpoint)
#     best_model = get_model(best_config).to(device)
#
#
#     best_model.load_state_dict(best_model_state)
#     best_model.eval()
#     test_pred = []
#     # test_logits=[]
#     with torch.no_grad():
#         for batch_x in test_loader:
#             batch_x = batch_x[0].to(device, dtype=torch.float)
#             _,logits = best_model(batch_x)
#             test_pred.append(logits.softmax(dim=1).cpu().numpy())
#     test_pred = np.concatenate(test_pred)
#     fold_pred.append(test_pred)
#
#     valid_loader=get_val_loader(X_valid)
#     valid_pred = []
#     with torch.no_grad():
#         for batch_x in valid_loader:
#             batch_x = batch_x[0].to(device, dtype=torch.float)
#             _,logits = best_model(batch_x)
#             valid_pred.append(logits.softmax(dim=1).cpu().numpy())
#     valid_pred = np.concatenate(valid_pred)
#     pseudo_labels.append(valid_pred)
#
# pseudo_labels_df=pd.DataFrame(np.concatenate(pseudo_labels,axis=0),columns=label_enc.classes_)
# pseudo_labels_df['Tour_ID']=np.concatenate(pseudo_ids,axis=0)
# test_predictions=np.stack(fold_pred).mean(axis=0)
#
#
# test_df=pd.DataFrame(test_predictions)
# test_df['Test_ID']=ids_test
# test_df_long=pd.melt(test_df,id_vars='Test_ID',var_name='category',value_name='prob')
# test_df_long['category']=label_enc.inverse_transform(test_df_long['category'].astype('int'))
# test_df_wide=pd.pivot_table(test_df_long,values='prob',index='Test_ID',columns='category')
#
# test_df_wide.to_csv(os.path.join(result_dir,"MLP semi-supervised.csv"))
# pseudo_labels_df.to_csv(os.path.join(result_dir,"Pseudo Labels - semi-supervised.csv"))
#
# loglosses=[f['logloss'] for f in kfold_results]
# print("losses:", loglosses)
# print("Mean loss: ",np.mean(loglosses))
