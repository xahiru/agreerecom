from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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
import pandas as pd
import os
import time
start=time.time()
print(start)

# reader = Reader(line_format='user item rating') #sep='\t',
# file_path = os.path.expanduser('~/.surprise_data/jester/jester_ratings.dat')
# data = Dataset.load_from_file(file_path, rating_scale=(-10, 10), reader=reader)
file_path_save_data = 'data/processed/'
# datasetname = 'ml-20m'
datasetname = 'ml-latest-small'
# datasetname = 'jester'

data = Dataset.load_builtin(datasetname)
# data = Dataset.load_builtin('ml-20m')
###

# trainset, testset = train_test_split(data, test_size=.2, train_size=None, random_state=100, shuffle=True)
# sim_options={'name':'pearson','user_based':False}
# algo = KNNWithMeans(sim_options=sim_options,verbose=True)
# algo.fit(trainset)
# predictions=algo.test(testset)
# rmse(predictions)
# mae(predictions)
# print("done")
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
beta = 2.5
if datasetname == 'jester':
    beta = 0


class MyOwnAlgorithm(AlgoBase):

    def __init__(self, k=40, min_k=1, alog=KNNWithMeans,user_based =True, beta=2.5, epsilon=0.9, lambdak=0.9, sim_options={}, verbose=True, **kwargs):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.k = k
        self.min_k = min_k
        self.algo = alog(k=k,sim_options=sim_options,verbose=True)
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

        # self.algo.means = np.zeros(self.n_x)
        # for x, ratings in iteritems(self.xr):
        #     self.algo.means[x] = np.mean([r for (_, r) in ratings])
        # self.sim = self.compute_similarities(verbose=self.verbose)
        self.algo.fit(trainset)
        # sim = self.algo.sim
        # tsim,activityma = agree_trust(trainset, self.beta, self.epsilon, ptype=self.ptype, istrainset=True, activity=False)

        # mixsim = (sim * tsim *tsim) 
        # self.algo.sim = mixsim
        print('Ignore the above similiary matrix generation message, its not used in this algorithm')

        tr, comon, noncom = agree_trust_opitmal_a_b(trainset, self.beta, self.epsilon, self.algo.sim, ptype=self.ptype, istrainset=True, activity=False)
        # self.algo.sim = tr*tr - noncom #works best for movie lens data sets
        # self.algo.sim = tr**0.5 - (0.6*noncom)#*tr*tr#*tr*tr*tr*tr*tr*tr*tr  #+ (noncom*noncom*noncom*noncom) no good for jester user based
        self.algo.sim = tr**self.lambdak - (self.epsilon*noncom)
        # self.algo.sim += 2*sim
        # self.algo.sim /= 2
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
        print('Ignore the above similiary matrix generation message, its not used in this algorithm')
        print('OdnovanAlgorithm here')
        start = time.time()
        if self.load == False:
            # self.algo.sim = odonovan_trust_old(trainset, self.algo, ptype=self.ptype, alpha=self.alpha)
            n_process = multiprocessing.cpu_count()
            self.algo.sim  = odonovan_trust_old(trainset, self.algo, ptype=self.ptype, alpha=self.alpha, optimized=True, n_jobs=n_process)
        print('OdnovanAlgorithm fit time')
        print('time.time() - start')
        print(time.time() - start)
        print('OdnovanAlgorithm fit done')
        print(self.algo.sim.shape)

    def estimate(self, u, i):
        # KNNWithMeans.test()
        # self.algo.estimate(u,i)

        return self.algo.estimate(u,i)


num_cores = multiprocessing.cpu_count()

# #### GRID search for parameter optimization
# user_based = True
# # param_grid = {'k': [10,20,30,40],'epsilon':[0,1,0.01,0.1,0.6,0.5,0.9,-1,-0.01,-0.1,-0.6,-0.5,-0.9], 'lambdak':[0.5,0.05,0.2,0.6,1,2,0.01,0.09] ,'user_based': [user_based], 'beta':[beta], 'sim_options': {'name': ['pearson'],
# param_grid = {'k': [40],'epsilon':[0,1,0.5,0.6], 'lambdak':[0.6,0.5] ,'user_based': [user_based], 'beta':[beta], 'sim_options': {'name': ['pearson'],
# # # #                        'epsilon':[00.7,0.6,0.9,0.5] 'lambdak':[0.5,0.05]  # 'min_support': [1, 5],
#                               'user_based': [user_based]}}
# # # param_grid = {'k': [40],'alog':[KNNWithMeans], 'alpha':[0.1,0.2,0.5,0.9], 'user_based': [True],'verbose':[False], 'sim_options': {'name': ['pearson'],
# #                               # 'min_support': [1, 5],
# #                               # 'user_based': [True]}}
# # # param_grid = {'k': [10,20,30,40],
# # #               'sim_options': {'name': ['pearson'],
# # #                               # 'min_support': [1, 5],
# # #                               'user_based': [user_based]}
# # #               }
# gs = GridSearchCV(MyOwnAlgorithm, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
# # # gs = GridSearchCV(OdnovanAlgorithm, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
# # # gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)

# gs.fit(data)

# # best RMSE score
# print(gs.best_score['rmse'])

# # combination of parameters that gave the best RMSE score
# print(gs.best_params['rmse'])

# print(gs.best_score['mae'])
# print(gs.best_params['mae'])

# results_df = pd.DataFrame.from_dict(gs.cv_results)
# results_df.to_csv('MyOwnAlgorithm_ml_latest_smalluser_based_True_gs_cv_results_2.csv', sep='\t', encoding='utf-8')


# #########
user_based = True
sim_options={'name':'pearson','user_based':user_based}
# cross_validate(MyOwnAlgorithm(k=40, alog=KNNWithMeans,user_based =False, beta=beta, epsilon=0, sim_options=sim_options), data, measures=['RMSE', 'MAE'], n_jobs=-1, cv=5, verbose=True)
# cross_validate(KNNWithMeans(k=40,sim_options=sim_options), data, measures=['RMSE', 'MAE'],n_jobs=-1, cv=5, verbose=True)
# cross_validate(OdnovanAlgorithm(alog=KNNWithMeansC, user_based=True, sim_options=sim_options, alpha=0.2), data, measures=['RMSE', 'MAE'] , cv=2, verbose=True)
# cross_validate(OdnovanAlgorithm(alog=KNNWithMeans, user_based=False, sim_options=sim_options, alpha=0.2), data, measures=['RMSE', 'MAE'] , cv=2, verbose=True)

# # # ##
kf = KFold(n_splits=5,  random_state=100)
alpha=0.2
predict_alog=KNNWithMeans
algo = OdnovanAlgorithm(alog=KNNWithMeans, sim_options=sim_options,load=False, user_based=user_based, alpha=alpha, verbose=False)
algo_name = 'OdnovanAlgorithm'
# epsilon=0.6
# epsilon=1
# lambdak=0.5
# predict_alog=KNNWithMeans
# algo = MyOwnAlgorithm(k=40, alog=predict_alog, user_based =user_based, beta=beta, epsilon=epsilon, lambdak=lambdak, sim_options=sim_options)
# algo_name = 'MyOwnAlgorithm'
# algo = KNNWithMeans(k=40,sim_options=sim_options)
# algo_name = 'KNNWithMeans'
# algo = SVD()
# algo_name = 'SVD'
# algo = KNNBasic(k=40,sim_options=sim_options)
# algo_name = 'KNNBasic'
# algo = KNNWithZScore()
# algo_name = 'KNNWithZScore'

sum_rmse = 0
sum_mae = 0
kt = 0

for trainset, testset in kf.split(data):
# # # #     # train and test algorithm.
#     algo.fit(trainset, testset)
    start = time.time()
    algo.fit(trainset)
    print(time.time() - start)
    # if kt == 0:
    if algo_name == 'MyOwnAlgorithm':
        np.save(datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_epsilon_'+str(epsilon)+'_lambdak_'+str(lambdak)+'_trust_matrix_.npy', algo.algo.sim)
    elif algo_name == 'OdnovanAlgorithm':
        if algo.load:
            algo.algo.sim = np.load(file_path_save_data+datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_alpha_'+str(alpha)+'_trust_matrix_.npy')
        else:
            np.save(file_path_save_data+datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_alpha_'+str(alpha)+'_trust_matrix_.npy', algo.algo.sim)
    else:
        np.save(file_path_save_data+datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_sim_matrix_.npy', algo.sim)
    # np.save(datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_trust_matix_.npy', algo.algo.sim)
   
    start = time.time()
    predictions = algo.test(testset)
    print(time.time() - start)

    # #     # Compute and print Root Mean Squared Error
    m_rmse = rmse(predictions, verbose=False)
    sum_rmse+= m_rmse
    m_mae = mae(predictions, verbose=False)
    sum_mae += m_mae
    kt += 1
    print('m_rmse')
    print(m_rmse)
    print('m_mae')
    print(m_mae)
    print(datasetname)

mean_mae = sum_mae/kt
mean_rmse = sum_rmse/kt
if algo_name == 'MyOwnAlgorithm':
    print(algo_name+'_predict_alog_'+str(predict_alog)+'_user_based_'+str(user_based)+'_epsilon_'+str(epsilon)+'_lambdak_'+str(lambdak))
elif algo_name == 'OdnovanAlgorithm':
    print(algo_name+'_predict_alog_'+str(predict_alog)+'_user_based_'+str(user_based)+'_alpha_'+str(alpha))
else:
    print(algo_name+'_user_based_'+str(user_based))
print('mean_rmse')
print(mean_rmse)
print('mean_mae')
print(mean_mae)

