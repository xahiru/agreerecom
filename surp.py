from __future__ import (absolute_import, division, print_function,             
                        unicode_literals)                                      
import pickle
import os

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time

from surprise import KNNBasic
from surprise import Dataset                                                     
from surprise import Reader                                                      
from surprise.accuracy import rmse
from surprise.accuracy import mae

from surprise import accuracy
from surprise.model_selection import train_test_split

from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import BaselineOnly

from sklearn.metrics.pairwise import pairwise_distances
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

# data.build_full_trainset()

trainset, testset2 = train_test_split(data, test_size=.2, train_size=None, random_state=100, shuffle=True)

# trainset = data.build_full_trainset()
testset = trainset.build_testset()
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
#                'user_based': False  # compute  similarities between items
            #    }
# # sim_options = {'name': 'pearson_baseline' ,
# #                'user_based': False  # compute  similarities between items
# #                }
# algo = KNNBasic(sim_options=sim_options)

# algo = KNNBasic()

is_user = False # if false item else user

if is_user == True:
    ptype = 'user'
else:
    ptype = 'item'

sim_options = {
    'name': 'pearson',
    'user_based': is_user
}
 
algo = KNNBasic(sim_options=sim_options)
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

        
        filtered_df = df.loc[df['err'] < 0.2] #alpha = 0.2

    
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


def agreement_nomal(trainset, alpha, ptype='user', istrainset=True):
    if istrainset == True:
        ratings = np.zeros((trainset.n_users, trainset.n_items))
        for u,i,r in trainset.all_ratings():
            ratings[u,i] =r
    else:
        ratings = trainset

    if ptype=='item':
        ratings = ratings.T


    trust_matrix = np.zeros((ratings.shape[0], ratings.shape[0]))
    
    for user_a in range(ratings.shape[0]):
            for user_b in range(ratings.shape[0]):
                if user_a != user_b:
                    a_ratings = ratings[user_a]
                    b_ratings = ratings[user_b]

                    commonset = np.intersect1d(np.nonzero(a_ratings), np.nonzero(b_ratings))
                    
                    common_set_length = len(commonset)

                    trust = 0

                    if(common_set_length > 0):
                        a_positive = a_ratings[commonset] > alpha
                        b_positive = b_ratings[commonset] > alpha

                        agreement = np.sum(np.logical_not(np.logical_xor(a_positive, b_positive)))

                        trust = agreement/common_set_length

                        # print(np.min(trust[np.nonzero(trust)]))

                    trust_matrix[user_a,user_b] = trust
            # print(trust_matrix[user_a])
    return trust_matrix
                    

def pitsmarsh_trust(trainset, algo, max_r, ptype='user'):
    ratings = np.zeros((trainset.n_users, trainset.n_items))
    for u,i,r in trainset.all_ratings():
        ratings[u,i] =r    

    if ptype=='item':
        ratings = ratings.T

    trainset2 = cp.deepcopy(trainset)
    algo.fit(trainset2)

    trust_matrix = np.zeros((ratings.shape[0], ratings.shape[0]))
    for a in range(ratings.shape[0]):
        for b in range(ratings.shape[0]):
            if (a!=b):
                r_a = ratings[a]
                r_b = ratings[b]

                common_index = np.intersect1d(np.nonzero(r_a),np.nonzero(r_b))
                    
                normalized_dif = sum(abs(r_a[common_index] - r_b[common_index])/max_r)

                common = sum(common_index)

                score = 0
                if(common != 0):
                    score = normalized_dif/common

                    # print(score)
                trust_matrix[a,b] = score
                    # print(a,b)
    # print(trust_matrix.shape)
    sim = algo.sim
    # similarity = 1 - pairwise_distances(ratings2, metric='cosine')
    # sim_backup = cp.deepcopy(similarity)
    # uncertainity = cp.deepcopy(trust_matrix)
    belief = (1 - trust_matrix) * (1 + sim)
    # disbilief = (1 - uncertainity) * (1 - sim_backup)
    # print(belief)
    return belief
    # return trust_matrix

def agreement_enhanced_on_estimate(trainset, algo, alpha, ptype='user', estrui='est'):
    print('======================== agreement_enhanced_on_estimate |START|========================')
    testset = trainset.build_testset() #trust is calculated for the same train n test set users
    # alpha = trainset.global_mean
    ratings = np.zeros((trainset.n_users, trainset.n_items))
    
    for u,i,r in trainset.all_ratings():
        ratings[u,i] =r    
    print(ratings.shape)


    if ptype == 'item':
        col_row_length = trainset.n_items
        sort_key = 'iid'
        print('trainset.ir')
        print(trainset.ir)
        keyset = trainset.ir
    else:
        col_row_length = trainset.n_users
        sort_key = 'uid'
        print('trainset.ur')
        print(trainset.ur)
        keyset = trainset.ur
    
    

    trust_matrix = np.zeros((col_row_length, col_row_length))
    p_trust_matrix = np.zeros((col_row_length, col_row_length))
    n_trust_matrix = np.zeros((col_row_length, col_row_length))
    # print('trainset.ur')
    # print(trainset.ur)
    print('trust_matrix.shape')
    print(trust_matrix.shape)
    print('======================== agreement_enhanced_on_estimate |Loop|========================')
    # for x in rang(1):
    for x in range(col_row_length):
        # print(trainset.ur[x])
        # print('x')
        # print(x)
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
        # df.sort_values(by=[sort_key])
        # df.sort_values(by=['iid'])
        # df.iid.astype(int)
        # print(df)
        # print(x)

        predictions = np.zeros((trainset.n_users, trainset.n_items))
        # print('predictions.shape')
        # print(predictions.shape)

        for t in p:
            predictions[trainset.to_inner_uid(t[0]), trainset.to_inner_iid(t[1])] = int(t[3]) # 3 is est, 2 is r_ui
            # print(t)
            # print(" 0 = "+str(t[0])+ " 1 = " +str(t[1])+ " 2 = " +str(t[3]))
        trust_matrix = trust_matrix + agreement_nomal(predictions, alpha, ptype, istrainset=False)
    
    print('======================== agreement_enhanced_on_estimate |END|========================')
    trust_matrix = trust_matrix / col_row_length
    return trust_matrix

start = time.time()
new_trust_matrix_od_item_pearson = gen_trust_matrix_leave_one_out(trainset,algo, testset, ptype='item')
# new_trust_matrix_od_item = gen_trust_matrix_leave_one_out(trainset,algo, testset, ptype='item')
# print(new_trust_matrix_od_user)
np.save('new_trust_matrix_od_item_pearson', new_trust_matrix_od_item_pearson)
# new_trust_matrix_od_user = np.load('new_trust_matrix_od_user.npy')
# np.save('new_trust_matrix_od_item3', new_trust_matrix_od_item)

# print('new_trust_matrix_od_user.shape')
# print(new_trust_matrix_od_user)
# # print('len(testset)')
# # print(len(testset))

# new_trust_matrix_agree_user = agreement_enhanced_on_estimate(trainset, algo, 2.5, ptype='item', estrui='est')
# new_trust_matrix_agree_item = agreement_enhanced_on_estimate(trainset, algo, testset, 2.5, ptype='item', estrui='est')
# np.save('new_trust_matrix_agree_user', new_trust_matrix_agree_user)
# new_trust_matrix_agree_user = np.load('new_trust_matrix_agree_user.npy')
# print('new_trust_matrix_agree_user')
# print(new_trust_matrix_agree_user)

# print(new_trust_matrix_agree_user)

# np.save('new_trust_matrix_agree_item3', new_trust_matrix_od_item)

# print('time taken to make trust matrices')
print(time.time() - start)
# plt.matshow(new_trust_matrix);
# plt.colorbar()
# plt.show()


###################################################################### uncommment followings for eveluation

algo.fit(trainset)



sim = algo.sim
sim2 = cp.deepcopy(sim)

p = algo.test(testset2)
print('normal')
rmse(p)
mae(p)


print('agreenormal_trust')
agreenormal_trust = agreement_nomal(trainset,2.5, ptype=ptype, istrainset=True)
agreenormal_trust_old = cp.deepcopy(agreenormal_trust)
algo.sim = agreenormal_trust_old
p = algo.test(testset2)
rmse(p)
mae(p)


# plt.matshow(agreenormal_trust);
# plt.colorbar()
# plt.show()

print('agreenormal_trust + sim)/2')
algo.sim = (agreenormal_trust + sim)/2
# # algo.sim = (2*(agreenormal_trust*sim))/(agreenormal_trust + sim)
p = algo.test(testset2)
rmse(p)
mae(p)


print('pitsmarsh_trust')
# algo.sim = pitsmarsh_trust(trainset, 5, ptype=ptype)
algo.sim = pitsmarsh_trust(trainset, algo, 5, ptype=ptype)
p = algo.test(testset2)
rmse(p)
mae(p)

print('odnovan_')
# algo.sim = pitsmarsh_trust(trainset, 5, ptype=ptype)
algo.sim = new_trust_matrix_od_item_pearson
p = algo.test(testset2)
rmse(p)
mae(p)

