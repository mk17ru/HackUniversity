%matplotlib inline
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

import pandas as pd

#import copy
#from google.colab import drive
#drive.mount("/content/drive")



df_train = pd.read_csv('drive/My Drive/added_train.csv', index_col='id')
df_test = pd.read_csv('drive/My Drive/added_test.csv', index_col='id')



y = df_train['ans']
X = df_train.drop(columns=['ans'])

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

model_cb = CatBoostClassifier(
    iterations = 15000,
    max_depth = 8).fit(X, y)
y_pred_cb = model_cb.predict_proba(X_test)[:, 1]

y_pred = (y_pred_cb > 0.5).astype(int)


df_submission = pd.DataFrame({'ans': y_pred}, index=df_test.index)

#df_submission.to_csv('submission_ensemble_cat_2.csv')
df_submission.to_csv('drive/My Drive/submission_ensemble_14000_8.csv')