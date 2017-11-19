import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import cPickle
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
import os
import shutil

if __name__ == '__main__':
    model_save = open('model.pkl', 'rb')
    model = cPickle.load(model_save)
    model_save.close()

    pandas2ri.activate()
    robjects.r.source('feature_extract.R')
    data = robjects.r('data')

    pred = model.predict(xgb.DMatrix(data), ntree_limit=model.best_ntree_limit)
    
    if pred < 0.5:
        sex = 'male'
    else:
        sex = 'female' 

    print(sex)

    names = os.listdir('data')
    file_read_path = 'data/'
    file_save_path = 'data_save/'
    for name in names:
        file_read = file_read_path + name
        file_save = file_save_path + name
        shutil.copyfile(file_read, file_save)
        os.remove(file_read)
