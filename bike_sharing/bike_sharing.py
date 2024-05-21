from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTreeRegression import DecisionTreeRegressor
import pandas as pd
import time
start_time = time.time()

bike_sharing_data = pd.read_csv('bike_sharing/bike_sharing.csv')
inputs = bike_sharing_data[['season','holiday','mnth','hr','temp','hum','windspeed','weekday','workingday','weathersit','yr','atemp']]
target = bike_sharing_data[['cnt']]

X_train, X_test, y_train, y_test = train_test_split(
    inputs, target, test_size=0.2, random_state=1234
)

clf = DecisionTreeRegressor(max_depth=4)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def mean_squared_error(y_test, y_pred):
    return np.mean((y_test - y_pred) ** 2)

def r_squared(y_test, y_pred):
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ss_res / ss_tot)

print(mean_squared_error(y_test.iloc[:,0], predictions))
print(r_squared(y_test.iloc[:,0], predictions))
print("--- %s seconds ---" % (time.time() - start_time))