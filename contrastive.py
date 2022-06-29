import os

import joblib
from pytorch_metric_learning import testers, miners, distances, reducers, losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_tabnet.tab_network import TabNet

import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from torch.optim import Adam
import torch.nn.functional as F

import ray
# ray.init("auto")
from modules import Classifier
from utils import permute_augmentation

ray.init( dashboard_host="0.0.0.0")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import log_loss
from sklearn.preprocessing import RobustScaler

import numpy as np
import pandas as pd

from settings import data_dir, result_dir, log_dir

margin=0.2
experiment=f"Contrastive-Supervised-{margin}"
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
        scl=RobustScaler()
        X[col]=scl.fit_transform(X[col].values.reshape(-1,1))
        X_test[col]=scl.transform(X_test[col].values.reshape(-1,1))



configs={
    'lr':tune.loguniform(0.0001,0.1),
    'l2': tune.loguniform(0.000001,0.001),
    'batch_size' : tune.choice([128]),
    # 'lambda_sparsity' : tune.loguniform(0.0000001,0.001),
    'emb_size':tune.choice([8,16,32]),
    'dim_hidden':tune.choice([1024,2048]),
    'n_hidden':tune.choice([1,2,]),
    'dropout':tune.uniform(0.01,0.2),

    'p_corrupt': tune.uniform(0.05,0.2),
    'dim_z':tune.choice([16,]),
    'margin':tune.choice([margin,]),
}
config={i:v.sample() for i,v in configs.items()}

def get_contrastive_model(config):
    model = Classifier(dim_x=X.shape[1], cat_idx=cat_idxs, cat_dims=cat_dims,
                       emb_size=config['emb_size'],n_hidden=config['n_hidden'],
                       dim_hidden=config['dim_hidden'],dropout=config['dropout'],
                       n_out=config['dim_z'])

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

def get_all_embeddings(dataset,model):
    tester=testers.BaseTester()
    return tester.get_all_embeddings(dataset,model)

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

def train_fun(model,criterion,optimizer,train_loader,val_loader,mining_func,accuracy_calculater,
              device='cpu',scheduler=None,p_corrupt=0.0):
    model.train()
    train_loss = 0
    total_train_loss=0
    train_z=[]
    train_labs=[]
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device,dtype=torch.float), batch_y.to(device,dtype=torch.int64)
        with torch.no_grad():
            batch_x=permute_augmentation(batch_x,p_corrupt=p_corrupt)
        embeddings = model(batch_x)
        embeddings = F.normalize(embeddings, dim=1)
        indices_tuple=mining_func(embeddings,batch_y)
        loss = criterion(embeddings, batch_y,indices_tuple)

        total_loss=loss+0
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader)
        total_train_loss += total_loss.item() / len(train_loader)
        train_z.append(embeddings)
        train_labs.append(batch_y)
    train_z=torch.cat(train_z,dim=0)
    train_labs=torch.cat(train_labs)

    model.eval()
    val_loss = 0
    total_val_loss=0
    val_z = []
    val_obs = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device,dtype=torch.float), batch_y.to(device,dtype=torch.int64)
            embeddings = model(batch_x)
            embeddings=F.normalize(embeddings,dim=1)
            indices_tuple = mining_func(embeddings, batch_y)
            loss = criterion(embeddings, batch_y, indices_tuple)
            total_loss = loss +0
            val_loss += loss.item() / len(val_loader)
            total_val_loss += total_loss.item()/len(val_loader)
            val_z.append(embeddings)
            val_obs.append(batch_y)

    val_z=torch.cat(val_z,dim=0)
    val_obs=torch.cat(val_obs,dim=0)
    accuracy=accuracy_calculater.get_accuracy(val_z,train_z,val_obs,train_labs,False)

    if scheduler: scheduler.step(val_loss)
    # val_pred = np.concatenate(val_pred)
    # val_obs = np.concatenate(val_obs)
    logloss =0# -discriminative_score(val_z.cpu().numpy(),val_obs.cpu().numpy())

    return train_loss,total_train_loss, val_loss,total_val_loss, logloss,accuracy['precision_at_1']

X_test = X_test.values
test_loader = get_val_loader(X_test)

pseudo_labels=[]
pseudo_ids=[]
X_train,X_valid,y_train,y_valid,ids_train,ids_valid=train_test_split(X,y,train['Tour_ID'].values,
                                                                     random_state=int(margin*1000),
                                                                     test_size=0.5)


X_train = X_train.values
X_valid = X_valid.values

class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_contrastive_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)
        distance=distances.CosineSimilarity()
        # distance=distances.DotProductSimilarity(normalize_embeddings=True)
        reducer=reducers.ThresholdReducer(low=0)
        self.mining_func=miners.TripletMarginMiner(margin=config['margin'],distance=distance,type_of_triplets='semihard')
        # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=50,gamma=0.5)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,
                                                                  patience=200,min_lr=1e-5)
        self.criterion=losses.TripletMarginLoss(margin=config['margin'],distance=distance,reducer=reducer)
        self.accuracy_calculater=AccuracyCalculator(include=("precision_at_1",),k=1)

        self.train_loader=get_train_loader(config,X_train,y_train)
        self.val_loader=get_val_loader(X_valid,y_valid)
        self.p_corrupt=config['p_corrupt']


    def step(self):
        train_loss, total_train_loss, val_loss, total_val_loss, logloss,accuracy \
            = train_fun(self.model, self.criterion,
                        self.optimizer, train_loader=self.train_loader,
                        val_loader=self.val_loader, device=device,
                        mining_func=self.mining_func,
                        accuracy_calculater=self.accuracy_calculater,
                        scheduler=self.scheduler, p_corrupt=self.p_corrupt)
        return {'train_loss':train_loss,'loss':val_loss,'logloss':logloss,
                'total_train_loss':total_train_loss,'total_val_loss':total_val_loss,
                'accuracy':accuracy}
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), checkpoint_path)
        return checkpoint_path
    def load_checkpoint(self, checkpoint_path):
        model_state,optimizer_state=torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

epochs=5000
scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=epochs,
        grace_period=400,
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
    local_dir=log_dir,
    num_samples=200,
    name=experiment,
    resume=False,
    scheduler=scheduler,
    # stop=early_stopper,
    progress_reporter=reporter,
    reuse_actors=False,
    raise_on_failed_trial=False)

metric='accuracy';mode='max';scope='last-5-avg'
best_trial = result.get_best_trial(metric, mode, scope=scope)
best_result = best_trial.last_result
print(best_result)

best_config = result.get_best_config(metric, mode, scope=scope)
best_checkpoint = result.get_best_checkpoint(best_trial, metric=metric, mode=mode)
best_model_state, _ = torch.load(best_checkpoint)
best_model = get_contrastive_model(best_config).to(device)


best_model.load_state_dict(best_model_state)
best_model.eval()
train_loader = get_val_loader(X_train)
train_z = []
with torch.no_grad():
    for batch_x in train_loader:
        batch_x = batch_x[0].to(device, dtype=torch.float)
        z = best_model(batch_x)
        train_z.append(F.normalize(z,dim=1).cpu().numpy())
train_z = np.concatenate(train_z)


test_z = []
with torch.no_grad():
    for batch_x in test_loader:
        batch_x = batch_x[0].to(device, dtype=torch.float)
        z = best_model(batch_x)
        test_z.append(F.normalize(z,dim=1).cpu().numpy())
test_z = np.concatenate(test_z)


valid_loader=get_val_loader(X_valid)
valid_z = []
with torch.no_grad():
    for batch_x in valid_loader:
        batch_x = batch_x[0].to(device, dtype=torch.float)
        z = best_model(batch_x)
        valid_z.append(F.normalize(z,dim=1).cpu().numpy())
valid_z = np.concatenate(valid_z)

params={'clf__C':[0.0001,0.005,0.001,0.05,0.01,0.05,0.1,0.5,1.0]}
base_clf=Pipeline([
    ('scl',StandardScaler()),
    ('clf',LogisticRegression(max_iter=10000))
])
clf=RandomizedSearchCV(base_clf,
                       param_distributions=params,
                       n_iter=20,scoring='neg_log_loss',
                       cv=5)

clf.fit(valid_z,y_valid)
# losses=[]
# models=[]
# for p in params:
#     clf=LogisticRegression(C=p,max_iter=10000)
#     clf.fit(valid_z,y_valid)
#     pred_train=clf.predict_proba(train_z)
#     loss_=log_loss(y_train,pred_train,labels=[0,1,2,3,4,5])
#     print(f"Parameter: {p}, loss {loss_:.3f}",)
#     losses.append(loss_)
#     models.append(clf)

# best_model_id=np.where(np.array(losses)==np.min(losses))[0][0]
# best_model=models[best_model_id]
test_predictions=clf.predict_proba(test_z)



joblib.dump(
    {'ids_train':ids_train,'ids_valid':ids_valid,'ids_test':test['Tour_ID'].values,
     'embeddings_train':train_z,'embeddings_valid':valid_z,'embeddings_test':test_z}
    ,os.path.join(result_dir,'Contrastive-Embeddings.joblib'))

torch.save((best_model_state,best_config),
            os.path.join(result_dir,"Contrastive-Embeddings.pth"))



test_df=pd.DataFrame(test_predictions)
test_df['Test_ID']=ids_test
test_df_long=pd.melt(test_df,id_vars='Test_ID',var_name='category',value_name='prob')
test_df_long['category']=label_enc.inverse_transform(test_df_long['category'].astype('int'))
test_df_wide=pd.pivot_table(test_df_long,values='prob',index='Test_ID',columns='category')

test_df_wide.to_csv(os.path.join(result_dir,f"{experiment}.csv"))


