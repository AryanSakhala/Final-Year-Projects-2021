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






Model  = pickle.load(open(r'E:\Projects2021\Telephone churning\static\models\model.sav', 'rb'))
le = LabelEncoder()
std = StandardScaler()
   
# def preprocess(test):
#     cat_cols=['Voice mail plan', 'International plan']
#     num_cols=['Account length',
#     'Number vmail messages',
#     'Total day minutes',
#     'Total day calls',
#     'Total eve minutes',
#     'Total eve calls',
#     'Total night minutes',
#     'Total night calls',
#     'Total intl minutes',
#     'Total intl calls',
#     'Customer service calls']
#     bin_cols=['International plan', 'Voice mail plan']

#     # for i in bin_cols:
#     #     x[i] = le.transform(x[i])

#     #comb = pd.get_dummies(data = x, columns = multi_cols)
    
#     scaled_test = std.fit_transform(test[num_cols])
#     scaled = pd.DataFrame(scaled_test, columns=num_cols)
#     test = test.drop(columns = num_cols, axis = 1)
#     test = test.merge(scaled, left_index=True, right_index=True, how = "left")
    
    
  
    
#     return test

def model_predict(Account_length,International_plan,Voice_mail_plan,Number_vmail_messages,Total_day_minutes,Total_day_call,Total_eve_minutes,Total_eve_calls,Total_night_minutes,Total_night_calls,Total_intl_minutes,Total_intl_calls,Customer_service_calls, model):
    x_train=[Account_length,International_plan,Voice_mail_plan,Number_vmail_messages,Total_day_minutes,Total_day_call,Total_eve_minutes,Total_eve_calls,Total_night_minutes,Total_night_calls,Total_intl_minutes,Total_intl_calls,Customer_service_calls]
    x_train=np.array(x_train)
    x_train=np.reshape(x_train,(-1,13))
    df=pd.DataFrame(x_train,columns=['International plan', 'Voice mail plan', 'Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls', 'Total eve minutes', 'Total eve calls', 'Total night minutes', 'Total night calls', 'Total intl minutes', 'Total intl calls', 'Customer service calls'] )
    # df=preprocess(df)
    preds = model.predict(df)
    #preds = np.argmax(preds, axis=1)
   
   

   
    if preds[0] == 0:
        pred = "No churn"
    else :
        pred = "Churn"
   

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
    print("Hello")
    if request.method == "POST":

        print("Hi")
        # state = request.form.get('state')
        Account_length = request.form.get('Account_length')
        # Area_code = request.form.get('Area_code')
        International_plan = request.form.get('International_plan')
        Voice_mail_plan = request.form.get('Voice_mail_plan')
        Number_vmail_messages = request.form.get('Number_vmail_messages')
        Total_day_minutes = request.form.get('Total_day_minutes')
        Total_day_calls = request.form.get('Total_day_calls')
        #Total_day_charge = request.form.get('Total_day_charge')
        Total_eve_minutes = request.form.get('Total_eve_minutes')
        Total_eve_calls = request.form.get('Total_eve_calls')
        #Total_eve_charge = request.form.get('Total_eve_charge')
        Total_night_minutes = request.form.get('Total_night_minutes')
        Total_night_calls = request.form.get('Total_night_calls')
        #Total_night_charge = request.form.get('Total_night_charge')
        Total_intl_minutes = request.form.get('Total_intl_minutes')
        Total_intl_calls = request.form.get('Total_intl_calls')
        #Total_intl_charge = request.form.get('Total_intl_charge')
        Customer_service_calls = request.form.get('Customer_service_calls')

        #print(float(Total_day_charge[:]))

        
        
        #TEST=preprocess(float(Total_day_charge[:]),float(Total_eve_charge[:]),float(Total_night_charge[:]),float(Total_intl_charge[:]),int(International_plan[:]),int(Customer_service_calls[:]))
        predict=model_predict(Account_length,International_plan,Voice_mail_plan,Number_vmail_messages,Total_day_minutes,Total_day_calls,Total_eve_minutes,Total_eve_calls,Total_night_minutes,Total_night_calls,Total_intl_minutes,Total_intl_calls,Customer_service_calls,Model)
        

        return render_template('prediction.html',prediction_text='Prediction {}'.format(predict))


app.run(debug=True)
