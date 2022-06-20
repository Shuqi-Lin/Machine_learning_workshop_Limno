#https://www.marksei.com/machine-learning-101-k-nearest-neighbors-classification-in-python/
"""
Example of KNN
"""
#%% Import the liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

#%%Read the inputs
df = pd.read_csv('Fish.csv')
#%% To visualize the first rows of the dataset
df.head()
df.info()

#%%to create a X variable containing all the features except "Species", and a y variable containing just the species. Both variables will only include observations about Perch and Bream species.
X = df[(df['Species'] == 'Perch') | (df['Species'] == 'Bream')].drop('Species', axis=1)
y = df[(df['Species'] == 'Perch') | (df['Species'] == 'Bream')]['Species'].replace(['Bream', 'Perch'], [0, 1])

#%%Split the dataset in two parts: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=101)
#%% Scale the parameters to the same limit
scaler = StandardScaler()           #determine the scaler
scaler = scaler.fit(X_train)        #fit on train phase 
X_train = scaler.transform(X_train) #tranform the scale on train set
X_test = scaler.transform(X_test)   #tranform the scale on test set
#%% Using KNN
knc = KNeighborsClassifier(2, p=2)
knc.fit(X_train, y_train)
print(classification_report(y_test, knc.predict(X_test)))


#%% Another Example to practice
X = df.drop(['Species'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=101) 

scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knc = KNeighborsClassifier(2, p=2)
knc.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=2, p=2,
                     weights='uniform')
print(classification_report(y_test, knc.predict(X_test), zero_division=0))
print(confusion_matrix(y_test, knc.predict(X_test)))