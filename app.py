import pickle
import re
from tkinter import Scale
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


#the starting point of my app from where it will been run
app=Flask(__name__)
##load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

#create the root for the home page
@app.route('/')
#create the home page
def home():
    return render_template('home.html')


#create a predict API
#give the input into the model and will give the output(pred)
@app.route('/predict_api',methods=['POST'])
def predict_api():
    #give the input in a json format wich will be captured inside the data key
    #give a post request with the information puted inside data in the json format
    #we capture it using request.json and we store it in the data var
    data=request.json['data']
    print(data)
    # we have to reshape the data respectfully to the that what we did in the model
    #a single data point record the model will get
    print(np.array(list(data.values())).reshape(1,-1))
    #scale the new data point 
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    #the output(prediction) in the model is a two dimensional array so so we take the first value
    print(output[0])
    return jsonify(output[0])


#create a post method
@app.route('/predict',methods=['POST'])
def predict():
    #take up all the values from the form
     data=[float(x) for x in request.form.values()] 
     #convert it to an array with the reshape and transform it with the scalar(standarizing the data)
     final_input=scalar.transform(np.array(data).reshape(1,-1))
     print(final_input)
     #predict and get the output
     output=regmodel.predict(final_input)[0]
     return render_template("home.html",prediction_text="The house price predicted is : {}".format(output))



if __name__=="__main__":
    #run the app in the debug mode
    app.run(debug=True)    




