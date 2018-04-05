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


from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
X_train, Y_train = x, y

from sklearn.preprocessing  import StandardScaler
sc=StandardScaler()
print(X_train.shape)

sc.fit(X_train)
X_train = sc.transform(X_train)

from sklearn.linear_model  import LogisticRegression
lr=LogisticRegression(C = 12500, random_state=215, max_iter=90, verbose = True)

lr.fit(X_train,Y_train)


# print(lr.coef_)
# print(lr.intercept_ )
# np.round(lr.predict_proba(X_test),3)

import pickle 
with open('train_model.pickle', 'wb') as f:
	pickle.dump(lr, f)