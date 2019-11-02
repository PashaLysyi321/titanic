import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')
idi = test['PassengerId'].valuesMetric
Your score is the percentage of passengers you correctly predict. This is known as accuracy.

Submission File Format
You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:

data.drop(['PassengerId', 'Cabin','Ticket','Name'], axis=1, inplace=True)
data = data.fillna(value = {'Age' : 23.79929, 'Embarked' : 'S','Fare' : 1000})
y_train = data['Survived'].values
data.drop(['Survived'], axis=1, inplace=True)

data = pd.get_dummies(data)
x_train = data

logReg = LogisticRegression(solver = 'lbfgs',penalty='l2')
logReg.fit(x_train, y_train)


test.drop(['PassengerId', 'Cabin','Ticket','Name'], axis=1, inplace=True)
test = test.fillna(value = {'Age' : 23.79929,'Fare' : 1000})
test = pd.get_dummies(test)

y = logReg.predict(test)
d = {'PassengerId':idi, 'Survived': y}
answer = pd.DataFrame(data=d)
np.savetxt(r'C:/Users/lysyi/Desktop/titanic/result.txt',answer,fmt='%d,%d')
