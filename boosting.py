import os

from catboost import CatBoostClassifier, Pool,sum_models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from scipy.special import softmax

import numpy as np
import pandas as pd

from settings import data_dir, result_dir, log_dir

experiment="Catboost"
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

cat_predictors=[ 'age_group', 'info_source',
       'tour_arrangement', 'package_transport_int', 'package_accomodation',
       'package_food', 'package_transport_tz', 'package_sightseeing',
       'package_guided_tour', 'package_insurance',  'first_trip_tz']

embedding_predictors=['country','travel_with','purpose','main_activity','info_source']


X_train=train[predictors].copy()
label_enc=LabelEncoder()
y_train=label_enc.fit_transform(train['cost_category'])

X_test=test[predictors].copy()
ids_test=test['Tour_ID']

# cat_features=[]
for i,col in enumerate(X_train.columns):
    if (col in cat_predictors) or (col in embedding_predictors):
        # cat_features.append(i)
        if col == 'country':
            mode="Others"
        else:
            mode=X_train[col].value_counts().index[0]
        # mode="Missing"
        X_train[col].fillna(mode,inplace=True)
        X_test[col].fillna(mode,inplace=True)
    else:
        median=X_train[col].median()
        X_train[col].fillna(median,inplace=True)
        X_test[col].fillna(median,inplace=True)


pool_train=Pool(X_train,y_train,cat_features=cat_predictors+embedding_predictors,)
pool_test=Pool(X_test,cat_features=cat_predictors+embedding_predictors,)


# model = CatBoostClassifier(loss_function='MultiClass',train_dir=os.path.join(log_dir,f'{experiment}'),
#                            verbose=False,task_type="GPU",
#                            # early_stopping_rounds=10
#                            )
#
# grid = {'learning_rate': [0.0005,0.001,0.005,0.01,0.05,0.1,0.3,],
#         'depth': [3,4,6,7,],
#         'l2_leaf_reg': [1, 3, 5, 7, 9],
#         # 'random_strength': [1,5,10,20],
#         'n_estimators':[500],
#         'bagging_temperature':[0.0,0.1,0.2,0.5,0.7,0.9,1.0]}
#
#
#
# grid_search_result = model.randomized_search(grid,
#                                        pool_train,
#                                        plot=False,
#                                        cv=10,
#                                         n_iter=300,
#                                              refit=False,
#                                              )

# best_params=model.get_params()
models = []
test_preds=[]
pseudo_labels=[]
pseudo_ids=[]
k_folds=KFold(n_splits=5,shuffle=True,random_state=123)
for train_ids,valid_ids in k_folds.split(X_train,y_train):
    X_train_fold=X_train.iloc[train_ids,:]
    X_valid_fold=X_train.iloc[valid_ids,:]
    y_train_fold=y_train[train_ids]
    y_valid_fold = y_train[valid_ids]
    pseudo_ids.append(train['Tour_ID'].iloc[valid_ids].values)
    pool_fold_train = Pool(X_train_fold, y_train_fold, cat_features=cat_predictors+embedding_predictors)
    pool_fold_valid = Pool(X_valid_fold,y_valid_fold, cat_features=cat_predictors+embedding_predictors)
    model_ = CatBoostClassifier(loss_function='MultiClass',
                                learning_rate=0.03,depth=6,
                                n_estimators=1000,
                                l2_leaf_reg=12,
                                bagging_temperature=1,
                                train_dir=os.path.join(log_dir,f'{experiment}'),
                           verbose=100,task_type="GPU",early_stopping_rounds=10)
    # model_.set_params(**best_params )
    model_.fit(pool_fold_train,
              eval_set=pool_fold_valid,
               use_best_model=True)
    models.append(model_)
    test_preds.append(model_.predict_proba(pool_test))
    pseudo_labels.append(model_.predict(pool_fold_valid,prediction_type="Probability"))

for m in models:
    print(m.best_score_," Trees:",m.tree_count_)
# models_avrg = sum_models(models,
#                          weights=[1.0/len(models)] * len(models))
#
# test_predictions_avg=models_avrg.predict(pool_test)
# test_predictions=softmax(test_predictions_avg,axis=1)

pseudo_labels_df=pd.DataFrame(np.concatenate(pseudo_labels,axis=0),columns=label_enc.classes_)
pseudo_labels_df['Tour_ID']=np.concatenate(pseudo_ids,axis=0)
test_predictions=np.stack(test_preds).mean(axis=0)

test_df=pd.DataFrame(test_predictions)
test_df['Test_ID']=ids_test
test_df_long=pd.melt(test_df,id_vars='Test_ID',var_name='category',value_name='prob')
test_df_long['category']=label_enc.inverse_transform(test_df_long['category'].astype('int'))
test_df_wide=pd.pivot_table(test_df_long,values='prob',index='Test_ID',columns='category')

test_df_wide.to_csv(os.path.join(result_dir,"Catboost baseline.csv"))
pseudo_labels_df.to_csv(os.path.join(result_dir,"Pseudo labels -boosting.csv"),index=False)
