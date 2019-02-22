
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


######################################### running parameters #############################
# file_path = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u1b.base'
# with open(file_path,'r') as f:
#     # next(f) # skip first row
#     df = pd.DataFrame(l.rstrip().split() for l in f)
import random
random.seed(300)


datasetname = 'ml-latest-small'
number_of_users = 10
file_path = os.path.expanduser('~') + '/.surprise_data/ml-latest-small/ratings.csv'
df = pd.read_csv(file_path) 
list = df.userId.unique()
random_number = random.randint(0,len(list)-number_of_users)
# list = df.movieId.unique()
# list = random.sample(set(list), 10) 
list = list[list[random_number:random_number+number_of_users]]
# df = df.loc[df['movieId'].isin(list)]
print(len(list))
print(list)
df = df.loc[df['userId'].isin(list)]

# print(df)
#reader is still required to load from dataframe
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']],rating_scale=(0.5, 5), reader=reader)

trainset = data.build_full_trainset()

beta = 2.5
if datasetname == 'jester':
    beta = 0


class MyOwnAlgorithm(AlgoBase):

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
        # sim = self.algo.sim
        # tsim,activityma = agree_trust(trainset, self.beta, self.epsilon, ptype=self.ptype, istrainset=True, activity=False)

        # mixsim = (sim * tsim *tsim) 
        # self.algo.sim = mixsim
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
    	# KNNWithMeans.test()
    	# self.algo.estimate(u,i)

        return self.algo.estimate(u,i)



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

# alpha=0.2
alpha=0.9
predict_alog=KNNWithMeans
algo = OdnovanAlgorithm(alog=KNNWithMeans, sim_options=sim_options,load=False, user_based=user_based, alpha=alpha, verbose=False)
algo_name = 'OdnovanAlgorithm'
# epsilon=1
# lambdak=0.5
# predict_alog=KNNWithMeans
# algo = MyOwnAlgorithm(k=40, alog=predict_alog, user_based =user_based, beta=beta, epsilon=epsilon, lambdak=lambdak, sim_options=sim_options, verbose=False)
# algo_name = 'MyOwnAlgorithm'
# algo = KNNWithMeans(k=40,sim_options=sim_options)
# algo_name = 'KNNWithMeans'
# algo = SVD()
# algo_name = 'SVD'
# algo = KNNBasic(k=40,sim_options=sim_options,verbose=True)
# algo_name = 'KNNBasic'
# algo = KNNWithZScore()
# algo_name = 'KNNWithZScore'
# k = 5
# for ktimes in range(k):
# start = time.time()
algo.fit(trainset)
testset = trainset.build_testset()
p = algo.test(testset)
rmse(p)
mae(p)
# print(time.time() - start)

######################################### grahps #############################
# trust_matrix = np.load('ml-20m0_OdnovanAlgorithm_user_based_True_alpha_0.2_trust_matrix_.npy')
# trust_matrix = np.load('ml-20m0_MyOwnAlgorithm_user_based_True_trust_matix_.npy')
# trust_matrix[trust_matrix > 1] = 1
# trust_matrix[trust_matrix < 1] = 0  

trust_matrix = algo.algo.sim
# trust_matrix = algo.sim
print(trust_matrix)

print(sum(x == 0 for row in trust_matrix for x in row))
# print(trust_matrix[30,40])
# print(trust_matrix[40,30])

# print(trust_matrix[37,54])
# print(trust_matrix[54,37])
plt.matshow(trust_matrix);
plt.colorbar()
plt.show()

