#!/usr/bin/env python
# coding: utf-8

# In[]:


# imports
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from datetime import datetime


# In[]:


# functions
def fk(X, Y, row, k=5):
    try:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X, Y)
        neighbors = knn.kneighbors([row], n_neighbors=k, return_distance=False)[0]
        result =  Y.iloc[neighbors].copy()
        result = result.append(Y.loc[[row.name]], ignore_index=False)
    except Exception as e:
        print('################## ERROR ##################')
        print('row.name: ', row.name)
        print('neighbors: ', neighbors)
        print('dataset size: ', len(X))
        raise e
    return result.value_counts(normalize=True)


def NEk(X, Y, k=5):
    S = X.copy()
    C = Y.copy()
    temp_sum = 0
    for index, row in S.iterrows():
        freqs = fk(S, C, row, k)
        for clss in set(C.values):
            try:
                freq = freqs.loc[clss]
                temp_sum += freq * np.log(freq)
            except:
                pass
    return (- 1/len(S)) * temp_sum


def NEFS(X, Y, k=5, s=2):
    if s > len(X.columns): raise Exception('s parameter must be less or equal to number of columns in X')
    elif s == len(X.columns): return X
    F = X.copy()
    C = Y.copy()
    S = pd.DataFrame()
    while not (len(S.columns) == s):
        nes = {}
        for col in list(F):
            Scopy = S.copy()
            Scopy[col] = F[col]
            nes[col] = NEk(Scopy, C, k)
        min_col = min(nes, key=nes.get)
        S[min_col] = F[min_col]
        F.drop(columns=[min_col], inplace=True)
    return S


# In[]:

# start = datetime.now()
# print(f'start: {start.strftime("%H:%M:%S")}')

# ds = pd.read_csv('cm1.csv').dropna()
# print(f'before removing duplicates: {len(ds)}')
# ds.drop_duplicates(inplace=True)
# print(f'after removing duplicates: {len(ds)}')

# X = ds.drop(columns=['label'])
# Y = ds.label
# # print(X.reset_index())

# main_2 = NEFS(X, Y, k=5, s=2)
# print(main_2.head())

# delta = datetime.now() - start
# print(f'end: {datetime.now().strftime("%H:%M:%S")}')
# print(f'duration: {delta} s')

