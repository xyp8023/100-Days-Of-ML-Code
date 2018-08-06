# preprocess the data
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# import the data
data_path = '../datasets/50_Startups.csv'
dataset = pd.read_csv(data_path)
X = dataset.iloc[ : , : -1].values # 0 R&D Spend 1 Administration 2 Market Spend 3 State
Y = dataset.iloc[ : , 4].values    # 4 profit
Y = np.array(Y).reshape(-1,1)

# #  handle the missing data  yet the result will not improve of using this trick
# X[:,0][X[:,0]==0.00]=np.nan
# X[:,1][X[:,1]==0.00]=np.nan
# X[:,2][X[:,2]==0.00]=np.nan
# imputer = Imputer(missing_values='NaN', strategy ='mean', axis=0)
# imputer = imputer.fit(X[:, 0:3])
# X[:, 0:3] = imputer.transform(X[:, 0:3])

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[ : ,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Standard Scale the data into 0 mean and unit variance
sc_X = StandardScaler()
sc_Y = StandardScaler()
X[:,3:6] = sc_X.fit_transform(X[:,3:6])
Y = sc_Y.fit_transform(Y)

# Avoid Dummy Variable Trap
X = X[: , 1:]

# split the dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#  fitting simple linear regression model to the training set
lr = LinearRegression()
lr.fit(X_train, Y_train)

#  predict the result
Y_pred = lr.predict(X_test)

# inverse the standard scale
Y_pred = sc_Y.inverse_transform(Y_pred)
Y_test = sc_Y.inverse_transform(Y_test)

#  visualization
plt.figure()
plt.plot(Y_test, color='blue')
plt.plot(Y_pred, color='red')
plt.show()

print('Done')