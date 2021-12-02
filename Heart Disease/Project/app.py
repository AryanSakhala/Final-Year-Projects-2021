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






Model  = pickle.load(open(r'E:\Projects2021\Heart Disease\Project\static\models\model.sav', 'rb'))


def model_predict(age,cp,trestbps,chol,fbs,restecg,thalachs,exang,oldpeak,slope,ca,thal, model):
    x_train=[age,cp,trestbps,chol,fbs,restecg,thalachs,exang,oldpeak,slope,ca,thal]
    x_train=np.array(x_train)
    x_train=np.reshape(x_train,(-1,12))
    df=pd.DataFrame(x_train,columns=['age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'] )
    
    preds = model.predict(df)
   
   
   

   
    if preds[0] == 0:
        pred = "Heart Disease"
    else :
        pred = "Healthy"
   

    return pred


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
        age = request.form.get('age')
        # Area_code = request.form.get('Area_code')
        cp = request.form.get('cp')
        trestbps = request.form.get('trestbps')
        chol = request.form.get('chol')
        fbs = request.form.get('fbs')
        restecg = request.form.get('restecg')
        #Total_day_charge = request.form.get('Total_day_charge')
        thalachs = request.form.get('thalachs')
        exang = request.form.get('exang')
        #Total_eve_charge = request.form.get('Total_eve_charge')
        oldpeak = request.form.get('oldpeak')
        slope = request.form.get('slope')
        #Total_night_charge = request.form.get('Total_night_charge')
        ca = request.form.get('ca')
        thal = request.form.get('thal')
        #Total_intl_charge = request.form.get('Total_intl_charge')
        

        #print(float(Total_day_charge[:]))

        
        
        #TEST=preprocess(float(Total_day_charge[:]),float(Total_eve_charge[:]),float(Total_night_charge[:]),float(Total_intl_charge[:]),int(International_plan[:]),int(Customer_service_calls[:]))
        predict=model_predict(age,cp,trestbps,chol,fbs,restecg,thalachs,exang,oldpeak,slope,ca,thal,Model)
        

        return render_template('prediction.html',prediction_text='Prediction {}'.format(predict))


app.run(debug=True)
