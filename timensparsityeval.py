
import numpy as np
from surprise import AlgoBase

import multiprocessing
from surprise import Dataset, evaluate
from surprise import Reader
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithMeansC
from collections import defaultdict
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.accuracy import rmse
from surprise.accuracy import mae
from surprise.model_selection import GridSearchCV
from surprise.agreements import agree_trust
from surprise.agreements import agree_trust_op
from surprise.agreements import odonovan_trust_old
from surprise.agreements import agree_trust_opitmal
from surprise.agreements import agree_trust_opitmal_a_b
from surprise.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp
import os
import time


######################################### running time parameters #############################

import random
random.seed(301)


datasetname = 'ml-latest-small'
number_of_users = 10
file_path = os.path.expanduser('~') + '/.surprise_data/ml-latest-small/ratings.csv'
df = pd.read_csv(file_path) 
list = df.userId.unique()
# list = df.movieId.unique()
print(len(list))
random_number = random.randint(0,len(list)-number_of_users)
# list = df.movieId.unique()
# list = random.sample(set(list), 10) 
list = list[list[random_number:random_number+number_of_users]]
# df = df.loc[df['movieId'].isin(list)]
print(len(list))
print(list)
df = df.loc[df['userId'].isin(list)]
# df = df.loc[df['rating'].isin([3])]

print(df)
#reader is still required to load from dataframe
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']],rating_scale=(0.5, 5), reader=reader)

trainset = data.build_full_trainset()

beta = 2.5
if datasetname == 'jester':
    beta = 0

###########################################################################AgreeTrust
class AgreeTrustAlgorithm(AlgoBase):

    def __init__(self, k=40, min_k=1, alog=KNNWithMeans,user_based =True, beta=2.5, epsilon=0.9, lambdak=0.9, sim_options={}, verbose=True, **kwargs):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.k = k
        self.min_k = min_k
        self.algo = alog(k=k,sim_options=sim_options,verbose=verbose)
        self.epsilon = epsilon
        self.lambdak =lambdak
        self.beta = beta
        if user_based:
            self.ptype = 'user'
        else:
            self.ptype = 'item'



    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        self.algo.fit(trainset)
        if self.algo.verbose:
        	print('Ignore the above similiary matrix generation message, its not used in this algorithm')
        print('Calculating AgreeTrust matrix ...')
        start = time.time()
        # tr, comon, noncom = agree_trust_opitmal_a_b(trainset, self.beta, self.epsilon, self.algo.sim, ptype=self.ptype, istrainset=True, activity=False)
        tr, comon, noncom = agree_trust_opitmal_a_b(trainset, self.beta, self.epsilon, self.algo.sim, ptype=self.ptype, istrainset=True, activity=False)
        self.algo.sim = tr**self.lambdak - (self.epsilon*noncom)
        # self.algo.sim[self.algo.sim > 1] = 1 
        print(time.time() - start)
        print('agree_trust_opitmal_a_b fit time')
        return self

    def estimate(self, u, i):

        return self.algo.estimate(u,i)


###########################################################################OdnovanAlgorithm
class OdnovanAlgorithm(AlgoBase):
    def __init__(self, k=40, min_k=1, alog=KNNWithMeans,user_based =True, alpha=0.2, sim_options={}, load=False, verbose=True, **kwargs):
        self.algo = alog(k=k,sim_options=sim_options,verbose=verbose)
        self.alpha = alpha
        self.load = load
        # self.testset = testset
        if user_based:
            self.ptype = 'user'
        else:
            self.ptype = 'item'
    def fit(self, trainset):
        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)
        self.algo.fit(trainset)
        if self.algo.verbose:
        	print('Ignore the above similiary matrix generation message, its not used in this algorithm')
        print('OdnovanAlgorithm here')
        start = time.time()
        if self.load == False:
        	n_process = multiprocessing.cpu_count()
        	self.algo.sim  = odonovan_trust_old(trainset, self.algo, ptype=self.ptype, alpha=self.alpha, optimized=True, n_jobs=n_process)
        print(time.time() - start)
        print('OdnovanAlgorithm fit time')
        print(self.algo.sim.shape)

    def estimate(self, u, i):
        # KNNWithMeans.test()
        # self.algo.estimate(u,i)

        return self.algo.estimate(u,i)


user_based = True
sim_options={'name':'pearson','user_based':user_based}

#### OdnovanAlgorithm (alog1)# ########################################################################
#### uncomment this section to run OdnovanAlgorithm, and comment other alogs

# alpha=0.01
# alpha=0.2
# predict_alog=KNNWithMeans
# algo = OdnovanAlgorithm(alog=KNNWithMeans, sim_options=sim_options,load=False, user_based=user_based, alpha=alpha, verbose=False)
# algo_name = 'OdnovanAlgorithm'

#### AgreeTrustAlgorithm (alog2)# ########################################################################
#### comment this section to run other alogrithms 
epsilon=1
lambdak=0.5
predict_alog=KNNWithMeans
algo = AgreeTrustAlgorithm(k=40, alog=predict_alog, user_based =user_based, beta=beta, epsilon=epsilon, lambdak=lambdak, sim_options=sim_options, verbose=False)
algo_name = 'AgreeTrustAlgorithm'

#### KNNBasic (alog3)# ########################################################################
#### uncomment this section to run KNNBasic, and comment other alogs
# algo = KNNBasic(k=40,sim_options=sim_options,verbose=True)
# algo_name = 'KNNBasic'

######################################### running time checking #############################
# start = time.time()
algo.fit(trainset)
# print(time.time() - start)

######################################### sparcity checking #############################

# trust_matrix = algo.algo.sim #for Odonovan and AgreeTrust algorithm
# # trust_matrix = algo.sim # for kNN basic uncomment this line commment above line
# print(trust_matrix)

# print(sum(x == 0 for row in trust_matrix for x in row))
