from __future__ import (absolute_import, division, print_function,             
                        unicode_literals)                                      
import pickle
import os

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time

from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset                                                     
from surprise import Reader                                                      
from surprise import dump
from surprise.accuracy import rmse
from surprise.accuracy import mae

from surprise import accuracy
from surprise.model_selection import train_test_split

from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import BaselineOnly

import copy as cp
# import random
# import matplotlib.pyplot as plt


# data = Dataset.load_builtin('ml-100k')

# https://github.com/NicolasHug/Surprise/blob/master/examples/notebooks/KNNBasic_analysis.ipynb
# https://surprise.readthedocs.io/en/v1.0.0/_modules/surprise/dataset.html
file_path = os.path.expanduser('~') + '/code/sci/recom/data/ml-100k/u1.base'

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

trainset, testset = train_test_split(data, test_size=.2, train_size=None, random_state=100, shuffle=True)

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

sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
# # sim_options = {'name': 'pearson_baseline' ,
# #                'user_based': False  # compute  similarities between items
# #                }
# algo = KNNBasic(sim_options=sim_options)

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



def gen_trust_matrix_leave_one_out(trainset, algo, testset, ptype='user'):
    print('======================== gen_trust_matrix_leave_one_out |START|========================')
    col_row_length = len(trainset.ur)
    
    if ptype == 'item':
        col_row_length = len(trainset.ir)

    trust_matrix = np.zeros((col_row_length, col_row_length))
    print('trust_matrix.shape')
    print(trust_matrix.shape)

    print('======================== gen_trust_matrix_leave_one_out |Loop|========================')
    # for x in range(1):
    for x in range(col_row_length):
        # print(trainset.ur[x])
        newset = cp.deepcopy(trainset)
        if ptype == 'user':
            newset.ur[x] = []
        else:
            newset.ir[x] = []
        
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
        # print('nu/den')
        # print(nu/den)

        trust_matrix[x,new_list] = nu/den
    
    print('======================== gen_trust_matrix_leave_one_out |END|========================')
    return trust_matrix

# Then compute RMSE
# accuracy.rmse(predictions)

# print(trainset.ur)

def agreement_enhanced_on_estimate(trainset, algo, testset, alpha, ptype='user', estrui='est'):
    print('======================== agreement_enhanced_on_estimate |START|========================')
    col_row_length = trainset.n_users
    
    if ptype == 'item':
        col_row_length = trainset.n_items
    
    # a1 = np.array(trainset.all_ratings())
    # a2 = np.array(trainset.all.[1])
    # A = np.matrix([a1,a2])
    # print(a1)


    trust_matrix = np.zeros((col_row_length, col_row_length))
    # print('trainset.ur')
    # print(trainset.ur)
    print('trust_matrix.shape')
    print(trust_matrix.shape)
    print('======================== agreement_enhanced_on_estimate |Loop|========================')
    # for x in range(1):
    for x in range(col_row_length):
        # print(trainset.ur[x])
        newset = cp.deepcopy(trainset)
        if ptype == 'user':
            a_row = cp.deepcopy(newset.ur[x])
            newset.ur[x] = []
        else:
            a_row = cp.deepcopy(newset.ir[x])
            newset.ir[x] = []
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
        # print('df')
        # print(df)

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
        # print('n_totals')
        # print(n_totals)
        
        new_list = [int(i)-1 for i in idx_positive_count]
        new_list2 = [int(j)-1 for j in idx_negative_count]

        positive_agreement =  positve_counts/p_totals
        # print('positive_agreement')
        # print(positive_agreement)
        negative_agreement = negatve_counts/n_totals
        # print('negative_agreement')
        # print(negative_agreement)
     
        trust_matrix[x,new_list] = positive_agreement
        trust_matrix_old = cp.deepcopy(trust_matrix)
        trust_matrix[x,new_list2] = negative_agreement
        trust_matrix = (trust_matrix_old + trust_matrix)/2
        # print('=========== agreement_enhanced_on_estimate |Loop|iteration|===========')
    
    print('======================== agreement_enhanced_on_estimate |END|========================')
    return trust_matrix

start = time.time()
new_trust_matrix_od_user = gen_trust_matrix_leave_one_out(trainset,algo, testset, ptype='user')
print('new_trust_matrix_od_user.shape')
print(new_trust_matrix_od_user)
# new_trust_matrix_od_item = gen_trust_matrix_leave_one_out(trainset,algo, testset, ptype='item')
# print(new_trust_matrix_od_user)
# np.save('new_trust_matrix_od_user3', new_trust_matrix_od_user)
# np.save('new_trust_matrix_od_item3', new_trust_matrix_od_item)
# # print('len(testset)')
# # print(len(testset))

new_trust_matrix_agree_user = agreement_enhanced_on_estimate(trainset, algo, testset, 2.5, ptype='user', estrui='est')
# new_trust_matrix_agree_item = agreement_enhanced_on_estimate(trainset, algo, testset, 2.5, ptype='item', estrui='est')
# np.save('new_trust_matrix_agree_user', new_trust_matrix_agree_user)
# new_trust_matrix_agree_user = np.load('new_trust_matrix_agree_user.npy')
print('new_trust_matrix_agree_user.shape')
print(new_trust_matrix_agree_user)

# print(new_trust_matrix_agree)

# np.save('new_trust_matrix_agree_item3', new_trust_matrix_od_item)

# print('time taken to make trust matrices')
print(time.time() - start)
# plt.matshow(new_trust_matrix);
# plt.colorbar()
# plt.show()



# new_trust_matrix_od = np.load('new_trust_matrix_od_item.npy')
# new_trust_matrix_agree_user = np.load('new_trust_matrix_agree_user.npy')
# new_trust_matrix_od_user = np.load('new_trust_matrix_od_user.npy')




algo.fit(trainset)

# print(algo.sim.shape)

# algo.sim = (2*(new_trust_matrix_od + algo.sim))/(new_trust_matrix_od + algo.sim)

# df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])


# sim = algo.sim

# print(sim)

# print(df)
p = algo.test(testset)
print('normal')
rmse(p)
mae(p)


algo.sim = new_trust_matrix_od_user
p = algo.test(testset)
print('new_trust_matrix_od_user')
rmse(p)
mae(p)

algo.sim = new_trust_matrix_agree_user
p = algo.test(testset)
print('new_trust_matrix_agree_user')
rmse(p)
mae(p)

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
#this is the important file
