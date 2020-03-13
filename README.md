# fraud-detection

ML-Model-Flask-Deployment
This is a demo project to predict credit card frued detection using Flask API

Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

Project Structure
This project has three major parts :

model.py - This contains code fot our Machine Learning model to predict if a transaction is fraud or not absed on trainign data in 'creditcard.csv' file.
app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.

templates - This folder contains the HTML template to allow user to enter transaction and predict if a transaction is fraud or not.
Running the project

Ensure that you are in the project home directory. Create the machine learning model by running below command -
python model.py

This would create a serialized version of our model into a file model.pkl

Run app.py using below command to start Flask API
python app.py

dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

heroku: https://cryptic-refuge-12933.herokuapp.com/
