# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

data=open("Qus_data.txt","r").read()
print(data)
import json
#to convert to json replace '' to ""
data=data.replace("'","\"")
data=json.loads(data)
print(data)

x=[]
y=[]
intent=list(data.keys())
print(intent)

for i in intent:
    for sample in data[i]:
        x.append(sample)
        y.append(intent.index(i))
        
print(x)
print(y)



import numpy as np
import spacy
nlp=spacy.load('en_core_web_sm')

corpus=[]
for i in range(len(x)):
    doc=nlp(x[i])
    review=[word.lemma_ for word in doc if not word.is_stop ]
    #lemma is data normalization, printing,print->print,drop duplicate word
    corpus.append(" ".join(review))
    
    
print(" corpus is ", corpus)

#only text cleaning is done-- text to vector still pending
#oneHot or tokenizer

from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

tok= Tokenizer(num_words=500)
tok.fit_on_texts(corpus)
print(corpus)
seq=tok.texts_to_sequences(corpus)
print(seq)

#length is diff, hence, padding

xdata = pad_sequences(seq, maxlen=4, padding='pre')
print(xdata)


from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical

model = Sequential()
model.add(Embedding(500,30,input_length=4))
model.add(LSTM(128,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(3,activation='softmax'))


#multiclass--actual label- oneHot encoded [0 1 2] to [001/010/100]
print(y)
ydata=to_categorical(y)
print(ydata)

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(xdata,ydata,epochs=100)


model.save(r'ChatBot.h5')
import joblib

joblib.dump(tok,r'tok.pkl')
