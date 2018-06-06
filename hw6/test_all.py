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
from sklearn.preprocessing import OneHotEncoder


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("Keras's version ----> ", keras.__version__)

Normalize = False
# mean, std = np.load("Mean&std.npy")
# print(mean, std)


## Load users.csv
User_info = [ [] for _ in range(6040)]
text = open(sys.argv[4] ,"r", encoding='ISO-8859-1')
for cou, line in enumerate(text):
	if cou != 0:
		if line.split("::")[1] == 'M' :
			User_info[int(line.split("::")[0])-1].append(int(0))
		else:
			User_info[int(line.split("::")[0])-1].append(int(1))
		User_info[int(line.split("::")[0])-1].append(int(line.split("::")[2]))
		User_info[int(line.split("::")[0])-1].append(int(line.split("::")[3]))
		# print(User_info[int(line.split("::")[0])-1])
		# input()
text.close()

print(np.asarray(User_info).shape)

print(User_info[0])
User_enc = OneHotEncoder()
User_info = User_enc.fit_transform(User_info).toarray()
print(User_info[0])

Dict = {'Animation': 0, "Children's": 1, 'Comedy': 2, 'Adventure': 3, 'Fantasy': 4,
		'Romance': 5, 'Drama': 6, 'Action': 7, 'Crime': 8, 'Thriller': 9, 'Horror': 10,
		'Sci-Fi': 11, 'Documentary': 12, 'War': 13, 'Musical': 14, 'Mystery': 15, 'Film-Noir': 16, 'Western': 17}


Movie_info = [ [] for _ in range(3952)]
## Load movies.csv
text = open(sys.argv[5] ,"r", encoding='ISO-8859-1')
for cou, line in enumerate(text):
	if cou != 0:
		tmp = line[:-1].split("::")[2].split("|")
		k = np.sum([to_categorical(Dict[item], num_classes=18)[0] for item in tmp], axis = 0)
		Movie_info[int(line.split("::")[0])-1]=k
text.close()

for i in range(len(Movie_info)):
	if Movie_info[i] == []:
		Movie_info[i] = np.asarray([0.]*18)

print(np.asarray(Movie_info).shape)



User, Movie = [], []
u_info, m_info = [], []

## Test Data Loading
text = open(sys.argv[1] ,"r", encoding='utf-8')
for cou, line in enumerate(text):
	if cou != 0:
		User.append(int(line.split(",")[1]))
		u_info.append(User_info[int(line.split(",")[1])-1])
		Movie.append(int(line.split(",")[2]))
		m_info.append(Movie_info[int(line.split(",")[2])-1])
text.close()
User, Movie = np.asarray(User), np.asarray(Movie)
n_users = np.amax(User)
n_movies = np.amax(Movie)
print(n_users, n_movies)
m_info, u_info = np.asarray(m_info), np.asarray(u_info)
print(m_info.shape, u_info.shape)





model = load_model(sys.argv[2])

print(model.summary())


print("---- Pridict... ----")
a = model.predict([User, Movie, u_info, m_info], verbose=1)
if Normalize:
	a = (a *std)+mean

print(a)
a = np.clip(a, 1, 5)
print(a)
print(a.shape)
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