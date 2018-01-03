#!/usr/bin/env python
#coding=utf-8
import cPickle as pickle
import sys
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split



reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ =='__main__':

	with open("embedding_1000.pkl", 'rb') as f:
		feature = pickle.load(f)

	with open("label_1000.pkl", 'rb') as f:
		label = pickle.load(f)

	print("Data load complete")

	x_train, x_test, y_train, y_test = train_test_split(feature, label, random_state=1, train_size=0.8)

	print("Start Training")

	clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovo')
	clf.fit(x_train, y_train)

	acc_train = clf.score(x_train, y_train)
	acc_test = clf.score(x_test, y_test)

	print("Train acc: ", acc_train)
	print("Test acc:", acc_test)

	with open("SVMmodel.pkl", "wb") as f:
		pickle.dump(clf, f)






