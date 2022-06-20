"""
Example of PCA
"""
#%% Import the liberariesimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%% Getting the inputs 
inp = pd.read_csv("pca.csv",header=0,index_col=0)
samples = inp[1:].values #TO remove NaN value in the first row
#%% Scale the parameters to the same limit
scaler = StandardScaler()                 #Create the scaler
normalizer=scaler.fit(samples)            #Fit them on samples
samples_norm=normalizer.transform(samples)#transform to samples 
#%%PCA
pca = PCA(n_components=2)                     #'n_components' is the number of the components that we want
components = pca.fit_transform(samples_norm)
X_new = pca.inverse_transform(components)

