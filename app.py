import numpy as np
import pandas as pd

import json

from flask import Flask, request, jsonify, render_template
import pickle
import requests



app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))




@app.route("/")
def index():
    return render_template("index.html")




@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.getlist('trans')
    predict_item = user_input[0]
    send_data = predict_item.split(",")
    send_data = list(map(float, send_data))
    send_data = np.array(send_data).reshape([1, -1])
    
    predictions = model.predict(send_data)
  


    
    
    return render_template('index.html', prediction_text=predictions)

    

if __name__ == "__main__":
	app.run(threaded=True, debug=True)