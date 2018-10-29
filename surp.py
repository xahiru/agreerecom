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
# import random


# data = Dataset.load_builtin('ml-100k')

# https://github.com/NicolasHug/Surprise/blob/master/examples/notebooks/KNNBasic_analysis.ipynb
# https://surprise.readthedocs.io/en/v1.0.0/_modules/surprise/dataset.html
# file_path = os.path.expanduser('~') + '/code/sci/recom/data/ml-100k/u.data'

file_path = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u.data'
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

# train_file = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u1.base'
# test_file = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u1.test'
# data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))

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
# random.seed(1)

trainset, testset = train_test_split(data, test_size=.20, train_size=None, random_state=100, shuffle=False)

# np.save('trainset.npy',trainset)
# np.save('testset.npy',testset)

# print(trainset)
# print(testset)

# trainset = np.load('trainset.npy')
# testset = np.load('testset.npy')

# sim_options = {'name': 'cosine',
#                'user_based': True  # compute  similarities between items
#                }
algo = KNNBasic()

# algo = MyOwnAlgorithm()
# algo = MyOwnAlgorithm()

# Train the algorithm on the trainset, and predict ratings for the testset
# trust_matix = np.load('data/ml-100k/agree/trust_matix_user.npy')



def gen_trust_matrix_leave_one_out(trainset, algo, testset):
    trust_matrix = np.zeros((len(trainset.ur), len(trainset.ur)))
    print('len(trainset.ur)')
   
    print(len(trainset.ur))
    print('trust_matrix.shape')
    print(trust_matrix.shape)

    for x in range(len(trainset.ur)):
        # print(trainset.ur[x])
        newset = cp.deepcopy(trainset)
        newset.ur[x] = []
        # print(len(newset.ur[x]))
        algo.fit(newset)
        p = algo.test(testset)
        # accuracy.rmse(p)
        df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])
        df.sort_values(by=['uid'])

        # df = df.head(100)

        df = df.loc[df['est'] != 0] #removes items predicted 0 
        df['err'] = abs(df.est - df.rui)

        
        filtered_df = df.loc[df['err'] < 0.2]

    
        uid1 = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts().keys().tolist()
        new_list = [int(i)-1 for i in uid1]
        # res = list(map(int, uid1))

        den = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts()

        uids = filtered_df.uid.value_counts().keys().tolist()
        nu = filtered_df.uid.value_counts()

        # print(sorted(uid1))
        # print(sorted(uids))

        trust_matrix[x,new_list] = nu/den

    return trust_matrix

# Then compute RMSE
# accuracy.rmse(predictions)

# print(trainset.ur)

new_trust_matrix = gen_trust_matrix_leave_one_out(trainset,algo, testset)
print(new_trust_matrix)
np.save('new_trust_matrix', new_trust_matrix)

# trainset.ur[2] = []


# algo.fit(trainset)


# sim = algo.sim

# # print(sim.shape)
# trust_matix = np.load('data/ml-100k/agree/trust_matix_user.npy')

# combined_sim_trust_user = (2*(trust_matix*sim))/(trust_matix + sim)

# # combined_sim_trust_user = (sim + trust_matix)/2

# algo.sim = combined_sim_trust_user

# # print(trust_matix.shape)

# predictions = algo.test(testset)

# accuracy.rmse(predictions)
# accuracy.mae(predictions)

# for trainset, testset in data.folds(): 
#     algo.train(trainset)                             
#     predictions = algo.test(testset)
#     rmse(predictions)
                                                                               
#     dump.dump('./dump_file', predictions, algo)
