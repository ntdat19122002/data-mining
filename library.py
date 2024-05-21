from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import pandas as pd
import time
start_time = time.time()

# game_data = pd.read_csv('mushroom_cleaned.csv')
# inputs = game_data[['cap-diameter','cap-shape','gill-attachment','gill-color','stem-height','stem-width','stem-color','season']]
# target = game_data[['class']]

game_data = pd.read_csv('chess/games.csv')
inputs = game_data[['rated','white_rating','black_rating','turns','moves']]
target = game_data[['winner']]
from sklearn.preprocessing import LabelEncoder
target = LabelEncoder().fit_transform(target)

# inputs.loc[:, 'increment_code'] = LabelEncoder().fit_transform(inputs['increment_code'])
inputs.loc[:,'rated'] = LabelEncoder().fit_transform(inputs['rated'])
inputs['delta_rate'] = inputs['white_rating'] - inputs['black_rating']
# inputs['first_move'] = inputs['moves'].apply(lambda x: x.split()[0])
# inputs.loc[:,'first_move'] = LabelEncoder().fit_transform(inputs['first_move'])
# inputs['second_move'] = inputs['moves'].apply(lambda x: x.split()[1])
# inputs.loc[:,'second_move'] = LabelEncoder().fit_transform(inputs['second_move'])
# inputs['third_move'] = inputs['moves'].apply(lambda x: x.split()[2])
# inputs.loc[:,'third_move'] = LabelEncoder().fit_transform(inputs['third_move'])
# inputs['fourth_move'] = inputs['moves'].apply(lambda x: x.split()[3])
# inputs.loc[:,'fourth_move'] = LabelEncoder().fit_transform(inputs['fourth_move'])
# inputs['fifth_move'] = inputs['moves'].apply(lambda x: x.split()[4])
# inputs.loc[:,'fifth_move'] = LabelEncoder().fit_transform(inputs['fifth_move'])
inputs.drop(columns=['moves'], inplace=True)

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