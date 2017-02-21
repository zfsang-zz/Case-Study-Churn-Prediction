import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sms
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix, precision_score, recall_score

#%matplotlib inline

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV



def load_data():
    odata = pd.read_csv('churn_train.csv')
    odata['last_trip_date'] = pd.to_datetime(odata['last_trip_date'],format='%Y-%m-%d')
    odata['signup_date'] = pd.to_datetime(odata['signup_date'],format='%Y-%m-%d')
    cutoff_date = datetime.strptime('2014-07-01','%Y-%m-%d').date() -pd.DateOffset(30, 'D')
    odata['active'] = odata['last_trip_date'] >= cutoff_date
    odata = pd.get_dummies(data=odata, columns=['city'])

    #take only required columns
    mdata = odata[['avg_dist','avg_surge','city_Astapor','city_King\'s Landing','city_Winterfell','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','active']]
    y = mdata.pop('active')
    X = mdata.values
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    return mdata,X_train,X_test,y_train,y_test

#jeff sang
def load_data_with_history_missing():
    odata = pd.read_csv('data/churn_test.csv')
    odata['last_trip_date'] = pd.to_datetime(odata['last_trip_date'],format='%Y-%m-%d')
    odata['signup_date'] = pd.to_datetime(odata['signup_date'],format='%Y-%m-%d')
    cutoff_date = datetime.strptime('2014-07-01','%Y-%m-%d').date() -pd.DateOffset(30, 'D')
    odata['history'] = (cutoff_date - odata['signup_date']).apply(lambda x:x/np.timedelta64(1,'D'))


    odata['active'] = odata['last_trip_date'] >= cutoff_date
    odata = pd.get_dummies(data=odata, columns=['city'])
    odata['Rating Greater than 4'] = odata['avg_rating_of_driver'] > 4
    # create history column
    # odata['history']
    # replace missing value by average value
    odata.fillna(odata.mean(),inplace=True)
    mdata = odata[['avg_dist','avg_rating_of_driver','avg_rating_by_driver','avg_surge','city_Astapor','city_King\'s Landing','city_Winterfell','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','history','active','Rating Greater than 4']]
    #mdata = mdata[mdata['city_Winterfell'] == 1]
    y = mdata.pop('active')
    X = mdata.values
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    return mdata, X_train,X_test,y_train,y_test


def standard_confusion_matrix(y_true, y_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def logistic_modeling(X_train, X_test, y_train, y_test):
    mod = LogisticRegression().fit(X_train, y_train)
    thre = 0.5
    predicted_y = mod.predict(X_test)
    MS = mean_squared_error(y_test, predicted_y)
    AS = accuracy_score(y_test, predicted_y)
    cm = standard_confusion_matrix(y_test, predicted_y)
    p = precision_score(y_test, predicted_y)
    r = recall_score(y_test, predicted_y)
    return mod, thre, MS, AS, cm, p, r

def GradientBoostingClassifier(X_train, X_test, y_train, y_test):
    mod = GBC(n_estimators=100,learning_rate=0.1,max_depth=5).fit(X_train,y_train)
    y_proba = mod.predict_proba(X_test)[:,1]
    thre = 0.5
    y_predict = (y_proba>thre).astype(int)
    MS = mean_squared_error(y_test,y_predict)
    AS = accuracy_score(y_test,y_predict)
    cm = standard_confusion_matrix(y_test, y_predict)
    p = precision_score(y_test, y_predict)
    r = recall_score(y_test, y_predict)
    return mod, thre, MS, AS, cm, p, r

def RandomForestClassifier(X_train, X_test, y_train, y_test):
    mod = RF(n_estimators=100,max_depth=5).fit(X_train,y_train)
    y_proba = mod.predict_proba(X_test)[:,1]
    thre = 0.5
    y_predict = (y_proba>thre).astype(int)
    MS = mean_squared_error(y_test,y_predict)
    AS = accuracy_score(y_test,y_predict)
    cm = standard_confusion_matrix(y_test, y_predict)
    p = precision_score(y_test, y_predict)
    r = recall_score(y_test, y_predict)
    return mod, thre, MS, AS, cm, p, r


if __name__ == '__main__':
    mdata, X_train, X_test, y_train, y_test = load_data_with_history_missing()
    print '------------Logistic Regression------------'
    # print 'Logistic Regression', result.summary()
    mod, thre, MS, AS, cm, p, r = logistic_modeling(X_train, X_test, y_train, y_test)
    print mod.coef_
    print 'Threshold : {} MeanSquaredError : {} Accuracy : {} Precision Score: {} Recall Score{}'.format(thre,MS,AS, p, r)
    print 'Confusion matrix', cm

    print '------------GradientBoostingClassifier------------'

    mod, thre, MS, AS, cm, p, r = GradientBoostingClassifier(X_train, X_test, y_train, y_test)
    print 'Threshold : {} MeanSquaredError : {} Accuracy : {} Precision Score: {} Recall Score{}'.format(thre,MS,AS, p, r)
    print 'Confusion matrix', cm

    print '------------RandomForestClassifier------------'

    mod, thre, MS, AS, cm, p, r = RandomForestClassifier(X_train, X_test, y_train, y_test)
    print 'Threshold : {} MeanSquaredError : {} Accuracy : {} Precision Score: {} Recall Score{}'.format(thre,MS,AS, p, r)
    print 'Confusion matrix', cm
    print '===================================================='

    '------------Feature Importance------------'
    feat_import = mod.feature_importances_
    top10_nx = np.argsort(feat_import)[::-1][0:14]
    feat_import = feat_import[top10_nx]
    print feat_import
    print mdata.columns[top10_nx]
