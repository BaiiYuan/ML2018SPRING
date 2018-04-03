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
	DF = np.concatenate(
		(DF, tmp**2, tmp**3, tmp**4, tmp**5, tmp**6,
			tmp**7, tmp**8, tmp**9, tmp**10, tmp**11,
			tmp**12, tmp**13, tmp**14, tmp**15, tmp**16,
			tmp**17, tmp**18,
			np.sin(tmp), np.cos(tmp), np.tan(tmp), np.arctan(tmp)),
		axis = 1)
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
lr=LogisticRegression(C = 5000, random_state=112, max_iter=100)

lr.fit(X_train,Y_train)


# print(lr.coef_)
# print(lr.intercept_ )
# np.round(lr.predict_proba(X_test),3)

from sklearn.externals import joblib
joblib.dump(lr, "train_model.m")