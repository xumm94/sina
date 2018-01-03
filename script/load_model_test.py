import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import pickle

if __name__ == '__main__':
    model_save = open('model.pkl', 'rb')
    model = pickle.load(model_save)
    model_save.close()

    test = pd.read_csv('test.csv')
    gender = {'male': 0, 'female': 1}
    test['label'] = test['label'].map(gender)
    cols = [c for c in test.columns if c not in ['label']]

    test_label = test['label']
    test_label = np.array(test_label).reshape([-1, 1])
    del (test['label'])

    pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

    pre_label = np.zeros([pred.shape[0], 1])
    for i in range(pred.shape[0]):
        if pred[i] >= 0.5:
            pre_label[i] = 1
        else:
            pre_label[i] = 0

    acc = np.mean(np.equal(pre_label, test_label).astype(np.float))
    print("测试集正确率： ", acc)