import sys
import csv
import pandas as pd
import numpy as np
import os
import keras
import re
from sklearn.preprocessing import StandardScaler
from keras.utils import *
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import gensim
# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("Keras's version ----> ", keras.__version__)
print("gensim's version ----> ", gensim.__version__)
SEN_LENGTH = 35;	print("SEN_LENGTH = ", SEN_LENGTH)
VEC_SIZE = 100;		print("VEC_SIZE = ", VEC_SIZE)
BATCH_COUNTER = 0;	print("BATCH_COUNTER = ", BATCH_COUNTER)
batch_size = 200;	print("batch_size = ", batch_size)

def Append_Zero(X):
	# print(np.asarray(X).shape)
	for i in range(len(X)):
		while len(X[i]) < SEN_LENGTH:
			X[i].append(np.asarray([0]*VEC_SIZE))
	return X
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
		'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
		'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
		'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
		'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
		'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
		'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
		'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
		'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
		'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
		'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
		'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
		'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
		'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
		'shouldn', 'wasn', 'weren', 'won', 'wouldn']


# print(re.sub('[^a-zA-Z!?]', ' ', "..qdk"))
def WordPreprocess(wordlist, tokenize = True):
	newwordlist = [re.sub(r'(.)\1+', r'\1', w) for w in wordlist]
	if tokenize:	
		newwordlist = [re.sub('[^a-zA-Z!?]', '', w) for w in newwordlist]
	else:
		newwordlist = [re.sub('[^a-zA-Z]', '', w) for w in newwordlist]
	newwordlist = [w for w in newwordlist if  w != '']
	# newwordlist = [w for w in newwordlist if w not in stopwords]
	# newwordlist = [w for w in newwordlist if (not len(w)==1) or ( w =='!' or w == '?')]
	return newwordlist
def generator(X):
	while True:
		global BATCH_COUNTER
		if BATCH_COUNTER+batch_size >= 200000:
			X_out = X[BATCH_COUNTER:]
			# Y_out = Y[BATCH_COUNTER:]
			BATCH_COUNTER = 0
		else:
			X_out = X[BATCH_COUNTER:BATCH_COUNTER+batch_size]
			# Y_out = Y[BATCH_COUNTER:BATCH_COUNTER+batch_size]
			BATCH_COUNTER += batch_size
		# print(np.asarray(X_out).shape)
		X_out_T = []
		for i in range(len(X_out)):	X_out_T.append([ np.asarray(vectors[X_out[i][j]]) for j in range(len(X_out[i]))])
		X_out_T = Append_Zero(X_out_T)
		yield (np.array(X_out_T))


X_test, tmp_x_test = [], []

## Test Data Loading
text = open(sys.argv[1] ,"r", encoding='utf-8')
for cou, line in enumerate(text):
	if cou != 0:		
		# tmp_x_test.append(line[:-1].split(',')[1].split()[:SEN_LENGTH])
		tmp_x_test.append(WordPreprocess(line[:].split())[:SEN_LENGTH])
		# tmp_x_test[cou-1].append('<p>')
		# print(tmp_x_test[cou-1]);input()
text.close()
print("     Test   Length --> ", len(tmp_x_test))


print("---- Word2Vec... ----")
word_model = Word2Vec.load(sys.argv[4])

vectors = word_model.wv

print("---- Fitting DATA... ----")
# print(tmp_x_test[1])

for i in range(len(tmp_x_test)):
	for j in range(SEN_LENGTH):
		tmp_x_test[i] = [w for w in tmp_x_test[i] if w in vectors.vocab]

# print(tmp_x_test[1])



print("---- Loading model... ----")
model = load_model(sys.argv[3])
# print(model.summary())

print("---- Predict... ----")

a = model.predict_generator(generator(tmp_x_test), steps=len(tmp_x_test)//batch_size, verbose=1,)
print(a.shape)
for x in range(10):
	print(a[x])
a = np.argmax(a,axis=1)
for x in range(10):
	print(a[x])
print("---- CSV writing... ----")
ans = []
for i in range(len(tmp_x_test)):
	ans.append([str(i)])
	ans[i].append(a[i])
	# if a[i] > 0.5:	ans[i].append(1)
	# else:			ans[i].append(0)
filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i])
	# print(ans[i])
text.close()
print("END")