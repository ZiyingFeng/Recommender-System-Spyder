#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 00:44:54 2019

@author: ziyingfeng
"""

import numpy as np
import pandas as pd
import heapq
from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise import accuracy
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Using Surprise package instead of Sklearn
# Because Surprise has its own SGD, therefore it can deal with sparse matrix
# For Sklearn, we need to fill the NAN with numbers it is not practical here

# %% Import data
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('ml-100k/u.data', sep = '\t', names = columns)
print(data.head())

# In Surprise package we don't need this part
# Here just generate a utility matrix for reference
utilityMatrix = pd.pivot_table(data, values = 'rating', 
                               index = 'user_id', 
                               columns = 'item_id', 
                               aggfunc = np.max)
print(utilityMatrix.shape)
movieUtilityMatrix = utilityMatrix.T
print(movieUtilityMatrix)

# %% Convert data
# Surprise has its own data input handle methods
# We need to use reader to read the dataframe and convert to trainset
# Since we are care doing item_based, care about similarity between items, we change the column order
reader = Reader(rating_scale = (1, 5))
item_based_data = Dataset.load_from_df(data[['item_id', 'user_id', 'rating']], reader)
full_trainset = item_based_data.build_full_trainset()
trainSet, testSet = train_test_split(item_based_data, test_size = 0.2, random_state = 10)

# %% Train and test for the missing ratings
# Model choosing NMF or SVD for matrix factorization
# random_state is seed so that the result could be repeated
model_NMF = NMF(n_factors = 15, random_state = 10)
model_SVD = SVD(n_factors = 15, random_state = 10)

model_NMF.fit(trainSet)
model_SVD.fit(trainSet)

# Use both models to predict the NAN ratings
# Check the RMSE of the predictions, RMSE is calculated based on the non-NAN value in the testSet
predictions_NMF = model_NMF.test(testSet)
predictions_SVD = model_SVD.test(testSet)
print("NMF model RMSE is {:.3f}".format(accuracy.rmse(predictions_NMF, verbose = False)))
print("SVD model RMSE is {:.3f}".format(accuracy.rmse(predictions_SVD, verbose = False)))

# %% Similarity matrix 
# Use the model in the whole dataset, and use similarity to recommend similar item (movie)
full_trainset = item_based_data.build_full_trainset()
model_NMF_fullTrainSet = NMF(n_factors = 15, random_state = 10)
model_SVD_fullTrainSet = SVD(n_factors = 15, random_state = 10)
model_NMF_fullTrainSet.fit(full_trainset)
model_SVD_fullTrainSet.fit(full_trainset)
simMatrix_NMF = model_NMF_fullTrainSet.compute_similarities() # default simularity method is MSD mean square difference
simMatrix_SVD = model_SVD_fullTrainSet.compute_similarities()
print('NMF similarity matrix shape: ', simMatrix_NMF.shape)
print('SVD similarity matrix shape: ', simMatrix_SVD.shape)

print('Print the first 7 rows and 7 columns of the NMF similarity matrix:')
print(simMatrix_NMF[0:6][0:6])
print('Print the first 7 rows and 7 columns of the SVD similarity matrix:')
print(simMatrix_SVD[0:6][0:6])

# %% Find similar items
items = pd.read_csv('ml-100k/u.item', sep = '|', header = None, encoding = 'latin-1')
items = items.iloc[:, 0:2]
items.columns = ['item_id', 'movie_title']
print('Print part of the item_id and the corresponding movie_title:')
print(items.iloc[0:9, :])
print('\n')
print('Select a movie that I like, and find its item_id')
print(items[items['movie_title'].str.contains('Star Wars')])

sim_selected = simMatrix_NMF[49]
print('\n', sim_selected)

# find the indices of the k most similar items with heap
k = 10
k_sim_idx_heap = []
for i in range(len(sim_selected)):
    if i == 49:
        continue
    if len(k_sim_idx_heap) < k:
        heapq.heappush(k_sim_idx_heap, (sim_selected[i], i))
    else:
        if sim_selected[i] > k_sim_idx_heap[0][0]:
            heapq.heappop(k_sim_idx_heap)
            heapq.heappush(k_sim_idx_heap, (sim_selected[i], i))
print(k_sim_idx_heap)

# print out the corresponding movie_title
print('\n The recommemed movies are:')
for i in range(len(k_sim_idx_heap)-1, -1, -1):
    idx = k_sim_idx_heap[i][1]
    print(items.iloc[idx, 1])

