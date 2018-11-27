from __future__ import (absolute_import, division, print_function,             
                        unicode_literals)                                      
import pickle
import os

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time

from surprise import KNNWithMeans
from surprise import Dataset                                                     
from surprise import Reader                                                      
from surprise.accuracy import rmse
from surprise.accuracy import mae

from surprise import accuracy
from surprise.model_selection import train_test_split

import copy as cp


######################################### loading data #############################
# https://surprise.readthedocs.io/en/v1.0.0/_modules/surprise/dataset.html
#change the file path to the data file
file_path = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u.data'
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

#testset2 is used for final evaluation
trainset, testset2 = train_test_split(data, test_size=.2, train_size=None, random_state=100, shuffle=True)
#test is used for building trust matrics for odonovan's method
testset = trainset.build_testset()

######################################### running parameters #############################
is_user = False # if false item else user
load_rustmatrix_from_file = False
save_models_to_file = False

max_r = 5
beta = max_r/2
alpha=0.2

if is_user == True:
    ptype = 'user'
else:
    ptype = 'item'
ptype_list = [ptype]

sim_options = {
    'name': 'pearson',
    'user_based': is_user
}
 
algo = KNNWithMeans(sim_options=sim_options, verbose=False)

######################################### trust modelling functions #############################

def odonovan_trust(trainset, algo, testset, ptype='user', alpha=0.2):
    """Computes knn version of trust matrix proposed by J. O’Donovan and B. Smyth, in “Trust in recommender systems,” """
    print('======================== odonovan_trust |START|========================')
    col_row_length = len(trainset.ur)
    
    if ptype == 'item':
        col_row_length = len(trainset.ir)

    trust_matrix = np.zeros((col_row_length, col_row_length))

    for x in range(col_row_length):
        newset = cp.deepcopy(trainset)
        if ptype == 'user':
            newset.ur[x] = []
        else:
            newset.ir[x] = []
        
        algo.fit(newset)
        p = algo.test(testset)

        df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])
        df.sort_values(by=['uid'])
        df = df.loc[df['est'] != 0] #removes items predicted 0 
        df['err'] = abs(df.est - df.rui)

        filtered_df = df.loc[df['err'] < alpha] #alpha = 0.2

        uid1 = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts().keys().tolist()
        new_list = [int(i)-1 for i in uid1]

        den = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts()

        uids = filtered_df.uid.value_counts().keys().tolist()
        nu = filtered_df.uid.value_counts()

        trust_matrix[x,new_list] = nu/den
    
    print('======================== odonovan_trust |END|========================')
    return trust_matrix

def agree_trust(trainset, beta, ptype='user', istrainset=True):
    print('======================== agree_trust |START|========================')
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
                        a_positive = a_ratings[commonset] > beta
                        b_positive = b_ratings[commonset] > beta

                        agreement = np.sum(np.logical_not(np.logical_xor(a_positive, b_positive)))

                        trust = agreement/common_set_length

                    trust_matrix[user_a,user_b] = trust
    print('======================== agree_trust |END|========================')
    return trust_matrix
                    

def pitsmarsh_trust(trainset, algo, max_r, ptype='user'):
    """Computes trust matrix proposed G. Pitsilis and L. F. Marshall, in 
    'A model of trust derivation from evidence for use in recommendation systems.' """
    print('======================== pitsmarsh_trust |START|========================')
    ratings = np.zeros((trainset.n_users, trainset.n_items))
    for u,i,r in trainset.all_ratings():
        ratings[u,i] =r    

    if ptype=='item':
        ratings = ratings.T

    trainset2 = cp.deepcopy(trainset)
    algo.fit(trainset2) #so that sim = algo.sim is available

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

                trust_matrix[a,b] = score

    sim = algo.sim
    belief = (1 - trust_matrix) * (1 + sim)
    print('======================== pitsmarsh_trust |END|========================')
    return belief

######################################### saving data #############################
def save_models(alog_list):
    trust_list = []
    sim = algo.sim
    for alogr in alog_list:
        if alogr == 'agree_trust':
            trust_matrix = agree_trust(trainset, beta, ptype=ptype, istrainset=True)
        elif alogr == 'sim_trust':
            trust_matrix = (agree_trust(trainset, beta, ptype=ptype, istrainset=True) + sim)/2
        elif alogr == 'pitsmarsh_trust':
            trust_matrix = pitsmarsh_trust(trainset, algo, max_r, ptype=ptype)
        elif alogr == 'odnovan_trust':
            trust_matrix = odonovan_trust(trainset,algo, testset, ptype=ptype, alpha=alpha)
        else:
            #KNN deafult
            algo.fit(trainset)
            trust_matrix = algo.sim
        #respective Folders should exists
        #example1 data/ml-100k/KNNwithMeans
        #example2 data/ml-100k/KNNwithMeans
        np.save('data/ml-100k/'+str(alogr)+'/trust_matix_'+str(ptype)+'.npy', trust_matrix)
        trust_list.append(trust_matrix)
    return trust_list

#########################################  eveluation #############################
def evalall(aloglist, trust_list=None):
    # sim = algo.sim #save for sim_trust
    if trust_list==None:
        sim = algo.sim
        for x in aloglist:
            if x == 'agree_trust':
                algo.sim = agree_trust(trainset, beta, ptype=ptype, istrainset=True)
            elif x == 'sim_trust':
                algo.sim = (agree_trust(trainset, beta, ptype=ptype, istrainset=True) + sim)/2
            elif x == 'pitsmarsh_trust':
                algo.sim = pitsmarsh_trust(trainset, algo, max_r, ptype=ptype)
            elif x == 'odnovan_trust':
                algo.sim = odonovan_trust(trainset,algo, testset, ptype=ptype, alpha=alpha)
            p = algo.test(testset2)
            rmse(p)
            mae(p)
    else:
        for x,t in zip(aloglist,trust_list):
            algo.sim = t
            p = algo.test(testset2)
            rmse(p)
            mae(p)

######################################### running eveluation #############################

aloglist = ['KNNWithMeans','agree_trust', 'sim_trust', 'pitsmarsh_trust','odnovan_trust']
algo.fit(trainset)

if load_rustmatrix_from_file == True:
    trust_list = []
    for alogr in aloglist:
        trust_matrix = np.load('data/ml-100k/'+str(alogr)+'/trust_matix_'+str(ptype)+'.npy')
        trust_list.append(trust_matrix) 
    evalall(aloglist, trust_list)
else:
    if save_models_to_file == True:
        evalall(aloglist, save_models(aloglist))
    else:
        evalall(aloglist)