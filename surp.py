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
from surprise import BaselineOnly

import copy as cp
# import random
import matplotlib.pyplot as plt


# data = Dataset.load_builtin('ml-100k')

# https://github.com/NicolasHug/Surprise/blob/master/examples/notebooks/KNNBasic_analysis.ipynb
# https://surprise.readthedocs.io/en/v1.0.0/_modules/surprise/dataset.html
file_path = os.path.expanduser('~') + '/code/sci/recom/data/ml-100k/u.data'

# file_path = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u.data'
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
# random.seed(100)

trainset, testset = train_test_split(data, test_size=.20, train_size=None, random_state=100, shuffle=True)

# np.save('trainset.npy', trainset.ur)
# np.save('testset.npy', testset)

# print(trainset)
# print(testset)

# trainset2 = np.load('trainset.npy')
# trainset2 = cp.deepcopy(trainset)
# print('trainset2')
# print(trainset2)
# print('trainset.ur[10]')
# print(trainset.ur)
# testset2 = np.load('testset.npy')

# print('trainset == trainset2')
# print(trainset.ur == trainset2)

# print('testset == testset2')
# print(testset == testset2)

# sim_options = {'name': 'cosine',
#                'user_based': True  # compute  similarities between items
#                }
algo = KNNBasic()
# bsl_options = {'method': 'als',
#                'n_epochs': 5,
#                'reg_u': 12,
#                'reg_i': 5
            #    }
# algo = BaselineOnly(bsl_options=bsl_options)


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

def agreement_enhanced_on_estimate(trainset, algo, testset, alpha, estrui='est'):
    trust_matrix = np.zeros((len(trainset.ur), len(trainset.ur)))
    print('len(trainset.ur)')
   
    print(len(trainset.ur))
    print('trust_matrix.shape')
    print(trust_matrix.shape)

    for x in range(len(trainset.ur)):
        # print(trainset.ur[x])
        newset = cp.deepcopy(trainset)
        a_row = cp.deepcopy(newset.ur[x])
        newset.ur[x] = []
        algo.fit(newset)
        p = algo.test(testset)
        # accuracy.rmse(p)
        df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])
        df.iid.astype(int)
        # df.uid.astype(int)
        # df.plot()
        # df.plot(x='uid', y='rui')
        # plt.show()
        
        df.sort_values(by=['uid'])

        # df = df.head(100)

        df = df.loc[df[estrui] != 0] #removes items predicted 0 

        a_idx = [int(row[0]) for row in a_row]

      
        commonset = df.loc[df.iid.astype(int).isin(np.intersect1d(df.iid.astype(int), a_idx))]


        
        idx_positive_count = commonset.loc[commonset[estrui] > 2.5].uid.value_counts().keys().tolist()
        positve_counts = commonset.loc[commonset[estrui] > 2.5].uid.value_counts()
        # print('positve_counts')
        # print(positve_counts)

        idx_negative_count = commonset.loc[commonset[estrui] < 2.5].uid.value_counts().keys().tolist()
        negatve_counts = commonset.loc[commonset[estrui] < 2.5].uid.value_counts()
        # print('negatve_counts')
        # print(negatve_counts)

        p_totals = commonset.loc[commonset.uid.astype(int).isin(idx_positive_count)].uid.value_counts()
        n_totals = commonset.loc[commonset.uid.astype(int).isin(idx_negative_count)].uid.value_counts()
        # print('p_totals')
        # print(p_totals)
        
        new_list = [int(i)-1 for i in idx_positive_count]
        new_list2 = [int(j)-1 for j in idx_negative_count]

        positive_agreement =  positve_counts/p_totals
        # print('positive_agreement')
        # print(positive_agreement)
        negative_agreement = negatve_counts/n_totals
     
        trust_matrix[x,new_list] = positive_agreement
        trust_matrix_old = cp.deepcopy(trust_matrix)
        trust_matrix[x,new_list2] = negative_agreement
        trust_matrix = (trust_matrix_old + trust_matrix)/2



    return trust_matrix

new_trust_matrix_od = gen_trust_matrix_leave_one_out(trainset,algo, testset)
# print(new_trust_matrix)
np.save('new_trust_matrix_od', new_trust_matrix_od)


new_trust_matrix_agree = agreement_enhanced_on_estimate(trainset, algo, testset, 2.5,'est')
# print(new_trust_matrix)
np.save('new_trust_matrix_agree', new_trust_matrix_agree)

# plt.matshow(new_trust_matrix);
# plt.colorbar()
# plt.show()



algo.fit(trainset)
p = algo.test(testset)
df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])


sim = algo.sim

print(sim)

print(df)

print(rmse(p))


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

