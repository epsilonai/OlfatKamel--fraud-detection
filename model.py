# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('dataset/creditcard.csv')


scaler = StandardScaler()
scaler2 = StandardScaler()
#scaling time
scaled_time = scaler.fit_transform(df[['Time']])
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
scaled_time = pd.Series(flat_list1)

#scaling the amount column
scaled_amount = scaler2.fit_transform(df[['Amount']])
flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount = pd.Series(flat_list2)

#concatenating newly created columns w original df
df = pd.concat([df, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)

#dropping old amount and time columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Splitting Data into Train and Test
X = df.drop('Class', 1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model

rf_model = RandomForestClassifier(
    n_estimators=1,
    criterion='gini',
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=5,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=16,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None).fit(X_train, y_train)


# Saving model to disk
pickle.dump(rf_model, open('rf_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('rf_model.pkl','rb'))



