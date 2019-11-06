import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import normalize
from sklearn import preprocessing

data = pd.read_csv('C:/Users/lysyi/Desktop/titanic/train.csv', sep=',')
test = pd.read_csv('C:/Users/lysyi/Desktop/titanic/test.csv', sep=',')

ans = data['Survived']
idi = test['PassengerId']

data.drop(columns=['PassengerId','Cabin','Ticket','Embarked','SibSp','Parch','Name','Survived'], inplace=True)
test.drop(columns=['PassengerId','Cabin','Ticket','Embarked','SibSp','Parch','Name'], inplace=True)

data['Pclass'] = data['Pclass'].astype(str)
test['Pclass'] = test['Pclass'].astype(str)

data = pd.get_dummies(data)
test = pd.get_dummies(test)
cols = data.columns

min_max_scaler = preprocessing.MinMaxScaler().fit_transform(data)
data_norm = pd.DataFrame(min_max_scaler, columns = cols)

min_max_scaler1 = preprocessing.MinMaxScaler().fit_transform(test)
test_norm = pd.DataFrame(min_max_scaler1, columns = cols)

data_norm = data_norm.fillna(value = {'Age' : data_norm['Age'].mean(), 'Fare': data_norm['Fare'].mean()})
test_norm = test_norm.fillna(value = {'Age' : test_norm['Age'].mean(), 'Fare': test_norm['Fare'].mean()})

model = RandomForestClassifier(n_estimators = 700, criterion='gini', bootstrap= True, max_depth= 20, max_features='sqrt', min_samples_leaf= 1, min_samples_split= 10, n_jobs=-1)
model.fit(data_norm, ans)

y = model.predict(test_norm)
d = {'PassengerId':idi, 'Survived': y}
answer = pd.DataFrame(data=d)
np.savetxt(r'C:/Users/lysyi/Desktop/titanic/result.txt',answer,fmt='%d,%d')