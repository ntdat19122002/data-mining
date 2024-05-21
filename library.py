from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import pandas as pd
import time
start_time = time.time()

game_data = pd.read_csv('mushroom_cleaned.csv')
inputs = game_data[['cap-diameter','cap-shape','gill-attachment','gill-color','stem-height','stem-width','stem-color','season']]
target = game_data[['class']]

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=4)

X_train, X_test, y_train, y_test = train_test_split(
    inputs, target, test_size=0.2, random_state=1234
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test.iloc[:,0].values == y_pred) / len(y_test)
    # return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)
print("--- %s seconds ---" % (time.time() - start_time))