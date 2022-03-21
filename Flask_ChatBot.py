# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 22:35:32 2022

@author: Monisha
"""

from flask  import Flask,request
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

model = load_model('ChatBot.h5')
tok= joblib.load(r'tok.pkl')


app= Flask(__name__)
answer = open(r"Ans_data.txt","r").read()
data=answer.replace("'","\"")
data=json.loads(data)
print(data)
intent=list(data.keys())
#with some feature input: client
@app.route('/', methods=['POST'])

def predict():
    #collect data
    out=request.get_json(force=True)
    out= out['key']
    print(out)
    seq=tok.texts_to_sequences([out])
    seq=pad_sequences(seq,maxlen=4)
    ans=model.predict(seq)
    out=np.argmax(ans)
    inn=intent[out]
    print(inn)
    ans=np.random.choice(data[inn])
    return str(ans)

app.run()