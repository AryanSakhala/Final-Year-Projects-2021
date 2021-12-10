import json
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, session, request, flash, redirect
import flask
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import StandardScaler
import sklearn.externals
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
with open('config.json', 'r') as c:
    params = json.load(c)["params"]
import numpy as np

app = Flask(__name__, template_folder='template', static_folder='static')
app.secret_key = 'super-secret-key'






Model  = pickle.load(open(r'model.sav', 'rb'))


def model_predict(sex, age, address, Medu, Fedu, traveltime, failures,
       paid, higher, internet, goout, G1, G2, model):
    x_train=[sex, age, address, Medu, Fedu, traveltime, failures,
       paid, higher, internet, goout, G1, G2]
    x_train=np.array(x_train)
    x_train=np.reshape(x_train,(-1,13))
    df=pd.DataFrame(x_train,columns=['sex', 'age', 'address', 'Medu', 'Fedu', 'traveltime', 'failures',
       'paid', 'higher', 'internet', 'goout', 'G1', 'G2'] )
    
    preds = model.predict(df)
   
   
   

   
    
   

    return preds[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == "POST":
        username = request.form.get("uname")
        userpass = request.form.get("pass")
        if username == params['admin_user'] and userpass == params['admin_password']:
            # set the session variable
            session['user'] = username
            return render_template("index.html", params=params)
    return render_template('login.html', params=params)


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    
    if request.method == "POST":

        
        # state = request.form.get('state')
        sex = request.form.get('sex')
        # Area_code = request.form.get('Area_code')
        age = request.form.get('age')
        address = request.form.get('address')
        Medu = request.form.get('Medu')
        Fedu = request.form.get('Fedu')
        traveltime = request.form.get('traveltime')
        #Total_day_charge = request.form.get('Total_day_charge')
        failures = request.form.get('failures')
        paid = request.form.get('paid')
        #Total_eve_charge = request.form.get('Total_eve_charge')
        higher = request.form.get('higher')
        internet = request.form.get('internet')
        #Total_night_charge = request.form.get('Total_night_charge')
        goout = request.form.get('goout')
        G1 = request.form.get('G1')
        G2 = request.form.get('G2')
        #Total_intl_charge = request.form.get('Total_intl_charge')
        

        #print(float(Total_day_charge[:]))

        
        
        #TEST=preprocess(float(Total_day_charge[:]),float(Total_eve_charge[:]),float(Total_night_charge[:]),float(Total_intl_charge[:]),int(International_plan[:]),int(Customer_service_calls[:]))
        predict=model_predict(sex, age, address, Medu, Fedu, traveltime, failures,
       paid, higher, internet, goout, G1, G2,Model)
        

        return render_template('prediction.html',prediction_text='Prediction {}'.format(predict))


app.run(debug=True)
