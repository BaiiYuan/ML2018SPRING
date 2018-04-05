import sys
import csv
import numpy as np
import pandas as pd
import math
def add_feature(DF):
	tmp = DF[['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']]
	print(tmp.shape)
	DF, tmp = np.asarray(DF), np.asarray(tmp)
	print(DF.shape)
	for i in range(2, 51):
		DF = np.concatenate((DF, tmp**i), axis = 1)
	DF = np.concatenate((DF, np.sin(tmp), np.cos(tmp), np.tan(tmp), np.arctan(tmp)), axis = 1)
	return DF

x = pd.read_csv(sys.argv[1])
# print(x.columns)
x = add_feature(x)
y = pd.read_csv(sys.argv[2], header = None)
y = np.ravel(y)
X_test = pd.read_csv(sys.argv[3])
X_test = add_feature(X_test)

X_train, Y_train = x, y

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# from sklearn.externals import joblib
# lr = joblib.load("train_model.m")

import pickle 
with open('./train_model.pickle', 'rb') as f:
    lr = pickle.load(f)

ans = []
a = lr.predict(X_test)
for i in range(len(X_test)):
	ans.append([str(i+1)])
	ans[i].append(a[i])
filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()