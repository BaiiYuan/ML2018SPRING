import sys
import csv
import pandas as pd
import numpy as np
import os
import keras
import re
from sklearn.preprocessing import StandardScaler
from keras.utils import *
from keras.layers import Input, Dense, LSTM, Flatten, Dropout, MaxPooling1D, AveragePooling1D, concatenate, GlobalMaxPooling1D, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Keras's version ----> ", keras.__version__)
SEN_LENGTH = 35;	print("SEN_LENGTH = ", SEN_LENGTH)
VEC_SIZE = 100;		print("VEC_SIZE = ", VEC_SIZE)
EPOCH = 60;			print("EPOCH = ", EPOCH)
LOAD_STATUS = False;	print("LOAD_STATUS = ", LOAD_STATUS)
BATCH_COUNTER = 0;	print("BATCH_COUNTER = ", BATCH_COUNTER)
batch_size = 150;	print("batch_size = ", batch_size)
# re.sub(r'(.)\1+', r'\1', "11223")

x, tmp_x_train, y = [], [], []
unlabel, tmp_unlable = [], []

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
def removeStopwords(wordlist, Stopwords):
    return [w for w in wordlist if w not in Stopwords]
# def Append_PAD(wordlist):
# 	for i in range(len(wordlist)):
# 		if len(wordlist[i]) < SEN_LENGTH:
# 			while len(wordlist[i]) < SEN_LENGTH:
# 				wordlist[i].append("<p>")
# 		else:
# 			wordlist[i] = wordlist[i][:SEN_LENGTH]
# 	return wordlist
def Append_Zero(X):
	# print(np.asarray(X).shape)
	for i in range(len(X)):
		while len(X[i]) < SEN_LENGTH:
			X[i].append(np.asarray([0]*VEC_SIZE))
	return X
def generator(X, Y):
	while True:
		global BATCH_COUNTER
		if BATCH_COUNTER+batch_size >= 180000:
			X_out = X[BATCH_COUNTER:]
			Y_out = Y[BATCH_COUNTER:]
			BATCH_COUNTER = 0
		else:
			X_out = X[BATCH_COUNTER:BATCH_COUNTER+batch_size]
			Y_out = Y[BATCH_COUNTER:BATCH_COUNTER+batch_size]
			BATCH_COUNTER += batch_size
		# print(np.asarray(X_out).shape)
		X_out_T = []
		for i in range(len(X_out)):	X_out_T.append([ np.asarray(vectors[X_out[i][j]]) for j in range(len(X_out[i]))])
		X_out_T = Append_Zero(X_out_T)
		yield (np.array(X_out_T), np.array(Y_out))
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

## Train Data Loading
text = open(sys.argv[1] ,"r", encoding='utf-8')
for cou, line in enumerate(text):
	tmp_x_train.append(WordPreprocess(line[10:-1].split(), tokenize= False)[:SEN_LENGTH])
	# tmp_x_train[cou].append('<p>')
	y.append(int(line[0]))
	# print(tmp_x_train);input()
text.close()
print("     Train  Length --> ", len(tmp_x_train))
y = np.asarray(y)

# Unlable Data Loading
# text = open(sys.argv[2] ,"r", encoding='utf-8')
# for cou, line in enumerate(text):
# 	tmp_unlable.append(WordPreprocess(line.split(), tokenize= False)[:SEN_LENGTH])
# 	# tmp_unlable[cou].append('<p>')
# 	# print(tmp_unlable);input()
# text.close()
# print("    Unlabel Length --> ", len(tmp_unlable))


print(tmp_x_train[1])

print("---- Word2Vec... ----")
# merge = tmp_x_train + tmp_unlable
# word_model = Word2Vec(merge, size=VEC_SIZE, window=5, min_count=5, workers=2)
# word_model.save("Word_test_nontoken.h5")
word_model = Word2Vec.load(sys.argv[4])
# word_model.save("Word_test2.h5")
# word_model = Word2Vec.load("Word_test2.h5")
vectors = word_model.wv


print("---- Fitting DATA... ----")
print(tmp_x_train[1])
for i in range(len(tmp_x_train)):
	for j in range(SEN_LENGTH):
		tmp_x_train[i] = [w for w in tmp_x_train[i] if w in vectors.vocab]

# for i in range(len(tmp_unlable)):
# 	for j in range(SEN_LENGTH):
# 		tmp_unlable[i] = [w for w in tmp_unlable[i] if w in vectors.vocab]

print(tmp_x_train[1])

# print("---- Appending PAD... ----")
# tmp_x_train = Append_PAD(tmp_x_train)
# tmp_x_test = Append_PAD(tmp_x_test)




print("---- Appending PAD... ----")

X_train, X_value_t, Y_train, Y_value = train_test_split(tmp_x_train, y, test_size = 0.1, random_state = 205)

X_value = []
for i in range(len(X_value_t)):	
	X_value.append([ np.asarray(vectors[X_value_t[i][j]]) for j in range(len(X_value_t[i]))])
	# input();print(X_value[i])
X_value = Append_Zero(X_value)
print(np.asarray(X_value).shape)



# create the model
if LOAD_STATUS:
	print("---- Loading model... ----")
	model = load_model(sys.argv[3])
else:
	print("---- RNN... ----")
	Input = Input(shape=(SEN_LENGTH, VEC_SIZE))
	# layer = Embedding(output_dim = 64, input_length=SEN_LENGTH)(Input)
	
	# layer = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(Input)
	# layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(layer)
	# layer = Bidirectional(LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.1))(layer)

	# layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(layer)
	# Test1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(layer)
	# Test2 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(layer)
	# layer = concatenate([Test1, Test2], axis=-1)
	# layer = Flatten()(layer)


	layer = LSTM(128, dropout=0.25, recurrent_dropout=0.2, return_sequences=False)(Input)
	# layer = LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=False)(Input)
	layer = Dense(512)(layer)
	layer = Dropout(0.5)(layer)
	
	layer = Dense(64)(layer)
	layer = Dropout(0.5)(layer)
	# layer = Dense(1, activation='sigmoid')(layer)
	layer = Dense(2, activation='softmax')(layer) 
	model = Model(inputs=Input, outputs=layer)
	adam = Adam(lr=4e-4)
	model.compile(
		# loss='binary_crossentropy',
		loss='categorical_crossentropy',
		optimizer=adam, metrics=['accuracy'])
	print(model.summary())


early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

Y_train = np_utils.to_categorical(Y_train, num_classes=2)
Y_value = np_utils.to_categorical(Y_value, num_classes=2)
print("ok")
# model.fit(X_train, Y_train, validation_data=(X_value, Y_value), epochs=EPOCH, batch_size=512)
train_history = model.fit_generator(generator(X_train, Y_train),
					steps_per_epoch=len(X_train)//batch_size,
					epochs=EPOCH,
					verbose=1,
					validation_data=(X_value, Y_value),
					callbacks=[early_stopping]
				)
model.save(sys.argv[3])