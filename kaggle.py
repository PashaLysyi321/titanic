import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint

data = pd.read_csv('C:/Users/lysyi/Desktop/titanic/train.csv', sep=',')
test = pd.read_csv('C:/Users/lysyi/Desktop/titanic/test.csv', sep=',')
idi = test['PassengerId'].values


data.drop(['PassengerId', 'Cabin','Ticket','Name'], axis=1, inplace=True)
data = data.fillna(value = {'Age' : 23.79929, 'Embarked' : 'S','Fare' : 1000})
y = data['Survived'].values
data.drop(['Survived'], axis=1, inplace=True)
data = pd.get_dummies(data)
data.drop(['Embarked_S','Embarked_Q','Embarked_C', 'Parch', 'Fare'], axis=1, inplace=True)


xg = RandomForestClassifier(max_features= 'sqrt')
param_grid = {
    'n_estimators': [150,100],
    'max_depth':[2,3,None],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ["gini", "entropy"]}

random_search = GridSearchCV(xg, param_grid=param_grid, cv=5, iid=False)

model = random_search.fit(data, y)

test.drop(['PassengerId', 'Cabin','Ticket','Name'], axis=1, inplace=True)
test = test.fillna(value = {'Age' : 23.79929,'Fare' : 1000})
test = pd.get_dummies(test)
test.drop(['Embarked_S','Embarked_Q','Embarked_C', 'Parch', 'Fare'], axis=1, inplace=True)

y = model.predict(test)
d = {'PassengerId':idi, 'Survived': y}
answer = pd.DataFrame(data=d)
np.savetxt(r'C:/Users/lysyi/Desktop/titanic/result.txt',answer,fmt='%d,%d')

