import os
import pandas as pd
from settings import data_dir


train=pd.read_csv(os.path.join(data_dir,'Train.csv'))
test=pd.read_csv(os.path.join(data_dir,'Test.csv'))




