# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:22:47 2022

@author: Monisha
"""

from tensorflow.keras import models

model =models.load_model(r'ChatBot.h5')


import joblib 
tok=joblib.load(r'tok.pkl')


#search ans based on intent
import json
import numpy as np
answer=open("Ans_data.txt","r").read()

data=answer.replace("'","\"")
data=json.loads(data)
print(data)
intent=list(data.keys())

from tensorflow.keras.preprocessing.sequence import pad_sequences

while True:
    qus = input("You : ")
    if(qus=='exit'):
        break
    #text to vector
    seq=tok.texts_to_sequences([qus])
    seq=pad_sequences(seq,maxlen=4)
    ans_1=model.predict(seq) 
    ans=np.argmax(ans_1,axis=1)
    print(ans)
    inn=intent[ans[0]]
    print(inn)
    ans= np.random.choice(data[inn])
    print("Bot : ", ans)
    