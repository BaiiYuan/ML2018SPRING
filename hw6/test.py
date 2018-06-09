import sys
import csv
import pandas as pd
import numpy as np
import os
import keras
from sklearn.preprocessing import StandardScaler
from keras.utils import *
from keras.layers import Input, Dense, Flatten, Dropout, merge, Embedding
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

print("Keras's version ----> ", keras.__version__)


User, Movie = [], []
## Test Data Loading
text = open(sys.argv[1] ,"r", encoding='utf-8')
for cou, line in enumerate(text):
	if cou != 0:
		User.append(int(line.split(",")[1]))
		Movie.append(int(line.split(",")[2]))
text.close()
User, Movie = np.asarray(User), np.asarray(Movie)
n_users = np.amax(User)
n_movies = np.amax(Movie)
print(n_users, n_movies)

model = load_model(sys.argv[2])

print(model.summary())

print("---- Pridict... ----")
a = model.predict([User, Movie], verbose=1)
# print(a)
a = np.clip(a, 1, 5)
# print(a)
# print(a.shape)
print("---- CSV writing... ----")
ans = []
for i in range(len(a)):
	ans.append([str(i+1)])
	ans[i].append(a[i][0])
filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["TestDataID","Rating"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()
print("End")