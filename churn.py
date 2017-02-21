import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot a
#%matplotlib inline

def load_data():
    odata = pd.read_csv('churn_train.csv')
    odata['last_trip_date'] = pd.to_datetime(odata['last_trip_date'],format='%Y-%m-%d')
    odata['signup_date'] = pd.to_datetime(odata['signup_date'],format='%Y-%m-%d')
    cutoff_date = datetime.strptime('2014-07-01','%Y-%m-%d').date() -pd.DateOffset(30, 'D')
    odata['active'] = odata['last_trip_date'] >= cutoff_date
    odata = pd.get_dummies(data=odata, drop_first=True, columns=['city'])

    #take only required columns
    mdata = odata[['avg_dist','avg_surge','city_King\'s Landing','city_Winterfell','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','active']]
    y = mdata.pop('active')
    X = mdata.values
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    return mdata,X_train,X_test,y_train,y_test
