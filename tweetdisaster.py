#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:20:44 2020

@author: seangao
"""

import pandas as pd
import re
import contractions
from emoticon_fix import emoticon_fix
from nltk.corpus import stopwords
import en_core_web_sm
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import LSTM, Dense, Dropout, GlobalMaxPool1D
from keras.callbacks import EarlyStopping

filepath = '/Users/seangao/Desktop/Research/disaster/nlp-getting-started/train.csv'

df_raw = pd.read_csv(filepath)

df = df_raw[['keyword', 'text', 'target']]

#KEYWORD CHECK
df['keyword'].fillna('no keyword', inplace=True)
lst_keyword = df['keyword'].to_list()
lst_keyword = list(set(lst_keyword))
df['keyword'] = df['keyword'].str.replace('%20', ' ')

#TEXT PREPROCESS
lst_text = df['text'].to_list()

lst_text = [re.sub(r'http\S+', '', x) for x in lst_text] #remove urls
lst_text = [re.sub(r'@\w+', '', x) for x in lst_text] #remove @s
lst_text = [contractions.fix(x) for x in lst_text] #fix contractions
lst_text = [emoticon_fix.emoticon_fix(x) for x in lst_text] #fix emoticons
lst_text = [re.sub(r'(\d),(\d)', r'\1\2', x) for x in lst_text] #fix thousand separator
lst_text = [re.sub('[^A-Za-z0-9]+', ' ', x) for x in lst_text] #remove punctuations
lst_text = [x.lower() for x in lst_text]

sp = en_core_web_sm.load()
stopwords = set(stopwords.words('english'))

def lemma(input_str):
    '''
    lemmatization and remove stopwords
    '''
    s = sp(input_str)
    
    input_list = []
    for word in s:
        w = word.lemma_
        input_list.append(w)
        
    output = ' '.join(word for word in input_list if word not in stopwords)
    return output

lst_text1 = []
for i in tqdm(lst_text):
    lst_text1.append(lemma(i))

lst_text1 = [re.sub(' +', ' ', x) for x in lst_text1]
lst_text1 = [x.strip() for x in lst_text1]

#VECTORIZE
embed_size = 100
maxlen = 100

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lst_text1)

X = tokenizer.texts_to_sequences(lst_text1)
X = pad_sequences(X, maxlen = maxlen)

y = df['target'].to_list()
#y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 42)

#GLOVE EMBEDDING
embedding_path = '/Users/seangao/Desktop/Research/GloVe/glove.6B/glove.6B.100d.txt'

embeddings_index = {}
f = open(embedding_path)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#MODEL BUILD & TRAIN
model = Sequential()
model.add(Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(64, return_sequences = True, dropout=0.1, recurrent_dropout=0.1)))
model.add(GlobalMaxPool1D())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1)
history = model.fit(X_train, y_train, 
                    validation_data=(y_test, y_test), 
                    batch_size=32, epochs=1000, callbacks=[es])

#TEST PREP
testfilepath = '/Users/seangao/Desktop/Research/disaster/nlp-getting-started/test.csv'

df_test = pd.read_csv(testfilepath)

lst_test = df_test['text'].to_list()

lst_test = [re.sub(r'http\S+', '', x) for x in lst_test]
lst_test = [re.sub(r'@\w+', '', x) for x in lst_test]
lst_test = [contractions.fix(x) for x in lst_test]
lst_test = [emoticon_fix.emoticon_fix(x) for x in lst_test]
lst_test = [re.sub(r'(\d),(\d)', r'\1\2', x) for x in lst_test]
lst_test = [re.sub('[^A-Za-z0-9]+', ' ', x) for x in lst_test]
lst_test = [x.lower() for x in lst_test]

lst_test1 = []
for i in tqdm(lst_test):
    lst_test1.append(lemma(i))

lst_test1 = [re.sub(' +', ' ', x) for x in lst_test1]
lst_test1 = [x.strip() for x in lst_test1]

X_pred = tokenizer.texts_to_sequences(lst_test1)
X_pred = pad_sequences(X_pred, maxlen = maxlen)

#SUBMISSION PREP
y_pred = model.predict(X_pred, verbose=1)
y_pred = np.where(y_pred > 0.5, 1, 0)

subfilepath = '/Users/seangao/Desktop/Research/disaster/nlp-getting-started/sample_submission.csv'

sub = pd.read_csv(subfilepath)

sub['target'] = y_pred

sub.to_csv('/Users/seangao/Desktop/Research/disaster/nlp-getting-started/submission.csv', index=False)
