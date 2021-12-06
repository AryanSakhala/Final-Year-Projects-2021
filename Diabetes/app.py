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






Model  = pickle.load(open(r'E:\Projects2021\Diabetes\modelRFC.sav', 'rb'))


def model_predict(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age, model):
    x_train=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    x_train=np.array(x_train)
    x_train=np.reshape(x_train,(-1,8))
    df=pd.DataFrame(x_train,columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'] )
    # df=preprocess(df)
    preds = model.predict(df)
    #preds = np.argmax(preds, axis=1)
   
   

   
    if preds[0] == 0:
        pred = "No diabetic"
    else :
        pred = "diabetic"
   

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
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = int(request.form.get('Glucose'))
        BloodPressure = int(request.form.get('BloodPressure'))
        SkinThickness = int(request.form.get('SkinThickness'))
        Insulin = int(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))
        print(type(Pregnancies))
        print(type(Glucose))
        print(type(Pregnancies))
        print(type(BMI))
        print(type(DiabetesPedigreeFunction))
     
        

       
        predict=model_predict(int(Pregnancies),Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Model)
        

        return render_template('prediction.html',prediction_text='Prediction {}'.format(predict))


app.run(debug=True)
