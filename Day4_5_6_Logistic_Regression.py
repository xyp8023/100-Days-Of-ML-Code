# preprocess the data
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# import the data
data_path = '../datasets/Social_Network_Ads.csv'
dataset = pd.read_csv(data_path)
print(dataset)
X = dataset.iloc[ : , : -1].values # 0 R&D Spend 1 Administration 2 Market Spend 3 State
Y = dataset.iloc[ : , 4].values    # 4 profit
Y = np.array(Y).reshape(-1,)

labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[ : ,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Standard Scale the data into 0 mean and unit variance
sc_X = StandardScaler()
X[:,2:5] = sc_X.fit_transform(X[:,2:5])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

X_test[:,2:5] = sc_X.inverse_transform(X_test[:,2:5])

# visualization
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax1.scatter(X_test[:,3], X_test[:,4],c=Y_test,s=25, marker='o', edgecolors='k', cmap=cm_bright)
ax1.set_xlabel('Age')
ax1.set_ylabel('EstimatedSalary')
ax1.set_title('Test Data')
fig1.savefig('../figures/Day456_ytest')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X_test[:,3], X_test[:,4],c=Y_pred,s=25, marker='o', edgecolors='k',cmap=cm_bright)
ax2.set_xlabel('Age')
ax2.set_ylabel('EstimatedSalary')
ax2.set_title('Prediction Data')
fig2.savefig('../figures/Day456_ypred')
plt.show()

print('Done')
