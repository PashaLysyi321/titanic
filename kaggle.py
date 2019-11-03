import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

data = pd.read_csv('C:/Users/lysyi/Desktop/titanic/train.csv', sep=',')
test = pd.read_csv('C:/Users/lysyi/Desktop/titanic/test.csv', sep=',')
idi = test['PassengerId'].values


data.drop(['PassengerId', 'Cabin','Ticket','Name'], axis=1, inplace=True)
data = data.fillna(value = {'Age' : 23.79929, 'Embarked' : 'S','Fare' : 1000})
y = data['Survived'].values
data.drop(['Survived'], axis=1, inplace=True)
data = pd.get_dummies(data)
data.drop(['Embarked_S','Embarked_Q','Embarked_C', 'Parch', 'Fare'], axis=1, inplace=True)

pipe = Pipeline([('classifier' , RandomForestClassifier())])
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10))}
]

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
best_clf = clf.fit(data, y)

test.drop(['PassengerId', 'Cabin','Ticket','Name'], axis=1, inplace=True)
test = test.fillna(value = {'Age' : 23.79929,'Fare' : 1000})
test = pd.get_dummies(test)
test.drop(['Embarked_S','Embarked_Q','Embarked_C', 'Parch', 'Fare'], axis=1, inplace=True)

y = best_clf.predict(test)
d = {'PassengerId':idi, 'Survived': y}
answer = pd.DataFrame(data=d)
np.savetxt(r'C:/Users/lysyi/Desktop/titanic/t.txt',answer,fmt='%d,%d')

