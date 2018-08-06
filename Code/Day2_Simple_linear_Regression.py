# 1.preprocess the data
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_path = '../datasets/Data.csv'
dataset = pd.read_csv(data_path)
X = dataset.iloc[ : , : 1].values
Y = dataset.iloc[ : , 1].values

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[ : , 0])


labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

# 2. fitting simple linear regression model to the training set
lr = LinearRegression()
lr.fit(X_train, Y_train)

# 3. predict the result
Y_pred = lr.predict(X_test)

# 4. visualization
plt.figure()
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.figure()
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, Y_pred, color='blue')
plt.show()

print('Done')
