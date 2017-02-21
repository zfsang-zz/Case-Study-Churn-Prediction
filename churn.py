import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sms
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
#%matplotlib inline

def load_data():
    odata = pd.read_csv('churn_train.csv')
    odata['last_trip_date'] = pd.to_datetime(odata['last_trip_date'],format='%Y-%m-%d')
    odata['signup_date'] = pd.to_datetime(odata['signup_date'],format='%Y-%m-%d')
    cutoff_date = datetime.strptime('2014-07-01','%Y-%m-%d').date() -pd.DateOffset(30, 'D')
    odata['active'] = odata['last_trip_date'] >= cutoff_date
    odata = pd.get_dummies(data=odata, columns=['city'])

    #take only required columns
    mdata = odata[['avg_dist','avg_surge','city_King\'s Landing','city_Winterfell','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','active']]
    y = mdata.pop('active')
    X = mdata.values
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    return mdata, X_train,X_test,y_train,y_test

# Jeff Li add

# def model_summary(X_train, X_test, y_train, y_test):
#     logit = sms.Logit(y_train, X_train)
#     result = logit.fit()
#     return result

def logistic_modeling(X_train, X_test, y_train, y_test):
    lr = LogisticRegression().fit(X_train, y_train)
    predicted_y = lr.predict(X_test)
    MSE = mean_squared_error(y_test, predicted_y)
    accuracy = accuracy_score(y_test, predicted_y)
    return MSE, accuracy

if __name__ == '__main__':
    mdata,X_train,X_test,y_train,y_test = load_data()

    # X_train, X_test, y_train, y_test = cross_val(X,y)
    # result = model_summary(X_train, X_test, y_train, y_test)

    '------------Logistic Regression------------'
    # print 'Logistic Regression', result.summary()
    MSE, accuracy = logistic_modeling(X_train, X_test, y_train, y_test)
    print 'Mean Squared Error', MSE
    print 'Accuracy', accuracy
