import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


iris = pd.read_csv('../input/Iris.csv')
iris.head()

# creating dependent and independent variables
X = iris.iloc[:,[1,2,3,4]]
y = iris.iloc[:,5]

# creating test and training sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# lets scale out variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set using euclidean distance 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# predicting using the trained classifier
y_pred = classifier.predict(X_test)

#looking at the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('the amount of right predictions is {} out of {}'.format(sum(y_test == y_pred), len(y_test)))



