# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:02:55 2022

@author: Shuqi Lin
"""
#%% Import the basic packages
import os #This module provides a portable way of using operating system dependent functionality
import pandas as pd # Basic library to handle dataframe
import numpy as np # The fundamental package for scientific computing 
import matplotlib.pyplot as plt # a comprehensive library for creating data visualizations
from scipy import interpolate
import time # This module provides various time-related functions, e.g. help you count the model running time

#%% Import required module from Scikit-learn       
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score,r2_score

#%% Load training dataset (for regression question)¶
# Make sure you are in the main folder('..\Machine_learning_workshop')
df=pd.read_csv('Regression dataset.csv',sep='\t',parse_dates=['Date'])

#%% Browse your dataset and create basic data visualizations
df.info()
n_var=int(input('How many variables to plot?'))
f,ax=plt.subplots(nrows=n_var,ncols=1,figsize=(12,4*n_var),sharex=True)
for i in range(n_var):
    var=input('Variable '+str(i+1)+':')
    df.plot(x='Date',y=var,ax=ax[i],style='bo',markersize=1.5)
    ax[i].set_ylabel(var)

#%% Choose the features and target, and make sure they are in the same length
X=df.dropna()[['SST', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow',
       'thermD', 'MLD', 'W']]
y=df.dropna()[['Chl']].values
print('Shape of feature dataframe: {}'.format(X.shape))
print('Length of target column {}'.format(len(y)))

#%% Split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Shape of training dataset: {}'.format(X_train.shape))
print('Shape of testing dataset: {}'.format(X_test.shape))

#%% Build the model with default setting
RF_model=RandomForestRegressor() # Stay with the default values for now
RF_model.fit(X_train,y_train)

print('The number of features: {}'.format(RF_model.n_features_))
print('\n')
print('Hyperparameters within the RF model: {}'.format(RF_model.get_params()))
print('\n')
# the coefficient of determination of the prediction.
print('R2 in training dataset: {}'.format(round(RF_model.score(X_train,y_train),2))) 

#%% Apply the model to the testing dataset
y_hat=RF_model.predict(X_test) # y_hat is the predictive values from the model
print('R2 in testing dataset: {}'.format(round(r2_score(y_test,y_hat),2))) 
# you can also use RF_model.score(X_test,y_test)

#%% Choose the hyperparameters for the model
RF_model=RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=8)
RF_model.fit(X_train,y_train)
print('R2 in training dataset: {}'.format(round(RF_model.score(X_train,y_train),2))) 
print('R2 in testing dataset: {}'.format(round(RF_model.score(X_test,y_test),2)))

param_grid = {'n_estimators':[int(x) for x in np.arange(100,500,50)],
         'max_depth':[int(x) for x in np.arange(2,10,2)],
         'min_samples_leaf':[int (x) for x in np.arange(2,10,2)]}

RSgrid = RandomizedSearchCV(estimator = RF_model,param_distributions = param_grid,
                            scoring='neg_mean_squared_error',n_iter = 10,cv = 5,verbose = 0,
                            random_state=101)
RSgrid.fit(X_train,y_train)
print(RSgrid.best_params_)

#%% Set the hyperparameters as the opimal parameter combination we got before
RF_model=RandomForestRegressor().set_params(**RSgrid.best_params_)
# Fit the model with training dataset
RF_model.fit(X_train,y_train)
print('R2 in training dataset: {}'.format(round(RF_model.score(X_train,y_train),2))) 
print('R2 in testing dataset: {}'.format(round(RF_model.score(X_test,y_test),2))) 

#%% Practical Use: Use RF model to fill the gaps in your observations
#%% Train the model with all known Chl in the original dataset
X_train=df.dropna()[['SST', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow','thermD', 'MLD', 'W']]
y_train=df.dropna()['Chl'].values
param_grid = {'n_estimators':[int(x) for x in np.arange(100,500,50)],
         'max_depth':[int(x) for x in np.arange(2,10,2)],
         'min_samples_leaf':[int (x) for x in np.arange(2,10,2)]}

RSgrid = RandomizedSearchCV(estimator = RF_model,param_distributions = param_grid,
                            scoring='neg_mean_squared_error',n_iter = 10,cv = 5,verbose = 0,
                            random_state=101)
RSgrid.fit(X_train,y_train)
RF_model=RandomForestRegressor().set_params(**RSgrid.best_params_)
RF_model.fit(X_train,y_train)
print('R2 in training dataset: {}'.format(round(RF_model.score(X_train,y_train),2)))

#%% Apply the model to the gaps in the original dataset
X_test=df.loc[df['Chl'].isna(),['SST', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow','thermD', 'MLD', 'W']]
yhat=RF_model.predict(X_test)
Pred=pd.DataFrame({'Pred':yhat},index=df.loc[df['Chl'].isna(),'Date'])

#%% Plot the prediction and observation
f,ax=plt.subplots(figsize=(12,4))
Pred.plot(style='ro',alpha=0.5,ax=ax)
df.plot(x='Date',y='Chl',style='go',ax=ax)
ax.set_xlim([pd.Timestamp(2004,1,1),pd.Timestamp(2008,1,1)])
ax.legend(['Prediction','Observation'],frameon=False)

#%%Load training dataset (for classification question)
# Make sure you are in the main folder('..\Machine_learning_workshop')
df=pd.read_csv('Classification dataset.csv',sep='\t',parse_dates=['Date'])

#%% Browse your dataset and create basic data visualizations
df.info()
fig,ax=plt.subplots(figsize = (12,2))
df[df['Bloom']==1].plot(x='Date',y='Bloom',style='rs',markersize=20,alpha=0.3,ax=ax)
ax.set_xlim([pd.Timestamp(2004,1,1),pd.Timestamp(2008,1,1)])
ax.legend('')
## Turn the target 'Bloom' column into categorical data
df['Bloom']=pd.Categorical(df['Bloom'])
## Check the number of days counted as algal bloom date
df['Bloom'].value_counts()
## 0 represents no bloom
## 1 represents bloom

#%% Start to built the RF classifier
# Choose the features and target, and make sure they are in the same length
X=df[['SST', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow',
       'thermD', 'MLD', 'W']]
y=df[['Bloom']].values
print('Shape of feature dataframe: {}'.format(X.shape))
print('Length of target column {}'.format(len(y)))
# Split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Shape of training dataset: {}'.format(X_train.shape))
print('Shape of testing dataset: {}'.format(X_test.shape))

#%% Since the classes in 'Bloom' are unbalanced, the class weight should be specified before training the model
RF_classifier=RandomForestClassifier(class_weight="balanced")
## When tuning the hyperparameters, the scoring method should be specified in terms of unbalanced classes
RSgrid = RandomizedSearchCV(estimator = RF_classifier,param_distributions = param_grid,
                            scoring='balanced_accuracy',n_iter = 10,cv = 10,verbose = 1,
                            random_state=101)

RSgrid.fit(X_train, y_train)
RF_classifier=RandomForestClassifier(class_weight="balanced").set_params(**RSgrid.best_params_)

#%% Train the model with the optimal hyperparameter combination
RF_classifier.fit(X_train, y_train)
print('R2 in training dataset: {}'.format(round(RF_classifier.score(X_train,y_train),2))) 
print('R2 in testing dataset: {}'.format(round(RF_classifier.score(X_test,y_test),2))) 

#%% Additional function in RF classifier: it gives predict class probabilities
Prob=pd.DataFrame(RF_classifier.predict_proba(X_test),columns=[0,1])
Prob['Pred']=RF_classifier.predict(X_test)
Prob.head(30)

#%% Practical Use: Assume we only have data in 1999-2018 and want to predict the algal bloom date in 2019-2020 based on the available environmental factors
train_df=df[df['Date']<pd.Timestamp(2019,1,1)]
test_df=df[df['Date']>=pd.Timestamp(2019,1,1)]
X_train=train_df[['SST', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow','thermD', 'MLD', 'W']]
y_train=train_df['Bloom']
RF_classifier=RandomForestClassifier(class_weight="balanced")
## When tuning the hyperparameters, the scoring method should be specified in terms of unbalanced classes
RSgrid = RandomizedSearchCV(estimator = RF_classifier,param_distributions = param_grid,
                            scoring='balanced_accuracy',n_iter = 10,cv = 10,verbose = 1,
                            random_state=101)

RSgrid.fit(X_train, y_train)
RF_classifier=RandomForestClassifier(class_weight="balanced").set_params(**RSgrid.best_params_)
RF_classifier.fit(X_train, y_train)
print('R2 in training dataset: {}'.format(round(RF_classifier.score(X_train,y_train),2))) 
#%% Apply the model to testing dataset
X_test=test_df[['SST', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow','thermD', 'MLD', 'W']]
y_test=test_df['Bloom'].values
Prob=pd.DataFrame(RF_classifier.predict_proba(X_test),columns=[0,1],index=test_df['Date'])
Prob['Pred']=RF_classifier.predict(X_test)
Prob['True']=y_test
Prob['True']=Prob['True'].astype('int64')
print('R2 in testing dataset: {}'.format(round(RF_classifier.score(X_test,y_test),2))) 

fig,ax=plt.subplots(figsize = (12,2))
Prob[Prob['Pred']==1].plot(y='Pred',style='bs',markersize=15,alpha=0.3,ax=ax,label='Predicted bloom')
(Prob[Prob['True']==1]+0.3).plot(y='True',style='rs',markersize=15,alpha=0.3,ax=ax,label='Observed bloom')
ax.set_xlim([pd.Timestamp(2019,1,1),pd.Timestamp(2021,1,1)])
ax.set_ylim([0.9,1.5])
ax.legend(ncol=2,frameon=False)

#%% Feature importance
importances = RF_classifier.feature_importances_
forest_importances = pd.Series(importances, index=X_test.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6,6))
forest_importances.plot.barh(ax=ax)
ax.set_title("Feature importances using mean decrease in impurity")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()