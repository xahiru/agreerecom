from __future__ import (absolute_import, division, print_function,             
                        unicode_literals)                                      
import pickle
import os

import pandas as pd
import numpy as np
from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset                                                     
from surprise import Reader                                                      
from surprise import dump
from surprise.accuracy import rmse

from surprise import accuracy
from surprise.model_selection import train_test_split

from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate

import copy as cp

data = Dataset.load_builtin('ml-100k')

# https://github.com/NicolasHug/Surprise/blob/master/examples/notebooks/KNNBasic_analysis.ipynb
# https://surprise.readthedocs.io/en/v1.0.0/_modules/surprise/dataset.html
# file_path = os.path.expanduser('~') + '/code/sci/recom/data/ml-100k/u.data'
# reader = Reader(line_format='user item rating timestamp', sep='\t')
# data = Dataset.load_from_file(file_path, reader=reader)

# class MyOwnAlgorithm(AlgoBase):

#     def __init__(self):

#         # Always call base method before doing anything.
#         AlgoBase.__init__(self)

#     def fit(self, trainset):

#         # Here again: call base method before doing anything.
#         AlgoBase.fit(self, trainset)
#         return self

#     def estimate(self, u, i):

#         return self.the_mean

# algo = SVD()

# algo.fit(trainset)
# predictions = algo.test(testset)

# # Then compute RMSE
# accuracy.rmse(predictions)

# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.20)

sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)

# algo = MyOwnAlgorithm()
# algo = MyOwnAlgorithm()

# Train the algorithm on the trainset, and predict ratings for the testset


# sim = algo.sim

def gen_trust_matrix_leave_one_out(trainset, batch_size, algo, testset):
    trust_matrix = np.zeros((batch_size, batch_size))

    for x in range(2):
        # print(trainset.ur[x])
        newset = cp.deepcopy(trainset)
        newset.ur[x] = []
        algo.fit(newset)
        p = algo.test(testset)
        df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])
        print(len(df.uid[df.uid.unique()]))

        print(abs(df.est - df.rui))
        

        # xhat_predict = algo.test(trainset)
        #     # print(np.any(np.isnan(xhat_predict)))

        # predic_diff = abs(prediction - xhat_predict)
        # predic_diff[np.isnan(predic_diff)] = 0
    
        #     # - (xhat_predict == 0).astype(bool).sum(dim) removes #of zero entries from numerator value
        # numerator = (xhat_predict < 0.2).astype(bool).sum(dim) - (xhat_predict == 0).astype(bool).sum(dim)
        #     # print(numerator)

        # denominator = xhat_predict.astype(bool).sum(dim)
        #     # print(denominator)
        # trust_row = numerator/denominator
        # trust_row[np.isnan(trust_row)] = 0
        # trust_row[np.isinf(trust_row)] = 0
            
        # trust_matrix[x] = trust_row
    return trust_matrix

# Then compute RMSE
# accuracy.rmse(predictions)

# print(trainset.ur)

new_trust_matrix = gen_trust_matrix_leave_one_out(trainset,10,algo, testset)

# trainset.ur[2] = []


# algo.fit(trainset)
# predictions = algo.test(testset)

# accuracy.rmse(predictions)
