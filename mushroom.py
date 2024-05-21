from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
from DecisionTreeC45 import DecisionTreeC45
import pandas as pd
import time
start_time = time.time()

game_data = pd.read_csv('mushroom_cleaned.csv')
inputs = game_data[['cap-diameter','cap-shape','gill-attachment','gill-color','stem-height','stem-width','stem-color','season']]
target = game_data[['class']]

X_train, X_test, y_train, y_test = train_test_split(
    inputs, target, test_size=0.2, random_state=1234
)

clf = DecisionTreeC45(max_depth=4)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test.iloc[:,0].values == y_pred) / len(y_test)
    # return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)
print("--- %s seconds ---" % (time.time() - start_time))