import json
import pickle
from flask import Flask,request,app, jsonify,url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

#load the model
regmodel=pickle.load(open('regmodel.pkl' , 'rb'))
scalar=pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api' , methods=['POST'])
def predict_api():
    data=request.json['data'] 
    # it will take all the data by making post api call & save in data
    print(data) # remember this is a json file
    print(np.array(list(data.values())).reshape(1,-1))
    # json will give dictionary{key:value} pairs , from there we will fetch values
    # now will convert those values to list ny doing list(data.values())
    # to feed this data in reg model we need 1 row containing all the features
    # so np.array(list(data.values())).reshape(1,-1) 
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data) # we will get a array with one entity we need 1st value
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(i) for i in request.form.values()]
    scaled_input=scalar.transform(np.array(data).reshape(1,-1))
    print(scaled_input)
    output1=regmodel.predict(scaled_input)[0]
    return render_template("home.html",prediction_text='the house price prediction comes around {}'.format(output1))


if __name__=="__main__":
    app.run(debug=True)
