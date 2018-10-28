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
trainset, testset = train_test_split(data, test_size=.20)

# sim_options = {'name': 'cosine',
#                'user_based': True  # compute  similarities between items
#                }
algo = KNNBasic()

# algo = MyOwnAlgorithm()
# algo = MyOwnAlgorithm()

# Train the algorithm on the trainset, and predict ratings for the testset
# trust_matix = np.load('data/ml-100k/agree/trust_matix_user.npy')



def gen_trust_matrix_leave_one_out(trainset, batch_size, algo, testset):
    trust_matrix = np.zeros((batch_size, batch_size))

    for x in range(1):
        # print(trainset.ur[x])
        newset = cp.deepcopy(trainset)
        newset.ur[x] = []
        # print(len(newset.ur[x]))
        algo.fit(newset)
        p = algo.test(testset)
        # accuracy.rmse(p)
        df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])
        # df.sort_index(by=['uid'], ascending=[True])
        df.sort_values(by=['uid'])

        # df = df.head(100)

        df = df.loc[df['est'] != 0] #removes items predicted 0 
        # print(df)
        # print(len(df.uid.unique()))
        # print(df.uid.value_counts())

        df['err'] = abs(df.est - df.rui)

        
        filtered_df = df.loc[df['err'] < 0.2]

        # filtered_df = filtered_df[['uid','iid']].groupby('uid')
        

        # filtered_df.apply(pd.Series.value_counts, axis=1)

        # print(len(df[filtered_df.uid.unique()].uid.value_counts()))
        # print(df.loc[df['uid'] == filtered_df.uid.unique()])
        # print(df.sort_values(by=['uid']))

        # print(df.loc[df['uid'].isin(filtered_df.uid.unique())])

        uid1 = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts().keys().tolist()
        den = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts().tolist()

        # print(df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts())
       
        # df['denominator'] = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts()
        
        # print(df.loc[df['uid'].isin(filtered_df.uid.unique())].uid)
        
        # print(filtered_df.uid.unique())

        uids = filtered_df.uid.value_counts().keys().tolist()
        nu = filtered_df.uid.value_counts().tolist()

        print(sorted(uid1))
        print(sorted(uids))

        # print(filtered_df.uid.value_counts())
        # df['numerator'] = filtered_df.uid.value_counts()
        # print(df.sort_values(by=['uid']))



        # print(numerator/denominator)

        # print(df.loc[filtered_df.uid.unique()])#numerator
        # print(filtered_df)#numerator
        # print(df.shape)
        # print(df.loc[filtered_df.uid.unique()])
    
        # print(abs(df.est - df.rui))

        # df.loc[df['column_name'] == some_value]
        

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
