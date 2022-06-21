import os

import pandas as pd
import numpy as np

from settings import result_dir

prediction_files=[
    'MLP baseline.csv',
    'MLP semi-supervised.csv',
    'Catboost baseline.csv',
    # 'Tabnet baseline.csv'
]

predictions=[]
for f in prediction_files:
    p=pd.read_csv(os.path.join(result_dir,f))
    # p['model']=f
    predictions.append(p)

predictions_long=pd.concat(predictions,axis=0)

predictions_final=predictions_long.groupby('Test_ID').agg(np.mean).reset_index()
predictions_final.to_csv(os.path.join(result_dir,"Stacking.csv"),index=False)
