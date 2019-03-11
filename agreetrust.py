from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from surprise import AlgoBase

import multiprocessing
from surprise import Dataset, evaluate
from surprise import Reader
from surprise import KNNWithMeans
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


file_path_save_data = 'data/processed/' #don't forget to create this folder before running the scrypt

datasetname = 'jester' #valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'

data = Dataset.load_builtin(datasetname)

###########################################################################
beta = 2.5
if datasetname == 'jester':
    beta = 0
user_based = True #changed to False to do item-absed CF
sim_options={'name':'pearson','user_based':user_based}

###########################################################################AgreeTrust
class AgreeTrustAlgorithm(AlgoBase):

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

        AlgoBase.fit(self, trainset)

        self.algo.fit(trainset)

        print('Ignore the above similiary matrix generation message, its not used in this algorithm')

        tr, comon, noncom = agree_trust_opitmal_a_b(trainset, self.beta, self.epsilon, self.algo.sim, ptype=self.ptype, istrainset=True, activity=False)
        self.algo.sim = tr**self.lambdak - (self.epsilon*noncom)
        return self

    def estimate(self, u, i):

        return self.algo.estimate(u,i)


###########################################################################OdnovanAlgorithm
class OdnovanAlgorithm(AlgoBase):
    def __init__(self, k=40, min_k=1, alog=KNNWithMeans,user_based =True, alpha=0.2, sim_options={}, load=False, verbose=True, **kwargs):
        self.algo = alog(k=k,sim_options=sim_options,verbose=verbose)
        self.alpha = alpha
        self.load = load
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
            
            n_process = multiprocessing.cpu_count()
            self.algo.sim  = odonovan_trust_old(trainset, self.algo, ptype=self.ptype, alpha=self.alpha, optimized=True, n_jobs=n_process)
        print('OdnovanAlgorithm fit time')
        print('time.time() - start')
        print(time.time() - start)
        print('OdnovanAlgorithm fit done')
        print(self.algo.sim.shape)

    def estimate(self, u, i):


        return self.algo.estimate(u,i)


num_cores = multiprocessing.cpu_count()

# ################################################# GRID search for parameter optimization ########
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
# gs = GridSearchCV(AgreeTrustAlgorithm, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
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
# results_df.to_csv('AgreeTrustAlgorithm_ml_latest_smalluser_based_True_gs_cv_results_2.csv', sep='\t', encoding='utf-8')





kf = KFold(n_splits=5,  random_state=100)
#### OdnovanAlgorithm (alog1)# ########################################################################
#### uncomment this section to run OdnovanAlgorithm, and comment other alogs
# alpha=0.2
# predict_alog=KNNWithMeans
# # if you want to load an already processed weight matrix then set load to True
## and it is set it at line 196
# algo = OdnovanAlgorithm(alog=KNNWithMeans, sim_options=sim_options,load=False, user_based=user_based, alpha=alpha, verbose=False)
# algo_name = 'OdnovanAlgorithm'

#### AgreeTrustAlgorithm (alog2)# ########################################################################
#### comment this section to run other alogrithms 
epsilon=1
lambdak=0.5
predict_alog=KNNWithMeans
algo = AgreeTrustAlgorithm(k=40, alog=predict_alog, user_based =user_based, beta=beta, epsilon=epsilon, lambdak=lambdak, sim_options=sim_options)
algo_name = 'AgreeTrustAlgorithm'

#### KNNBasic (alog3)# ########################################################################
#### uncomment this section to run KNNBasic, and comment other alogs
# algo = KNNBasic(k=40,sim_options=sim_options)
# algo_name = 'KNNBasic'



sum_rmse = 0
sum_mae = 0
kt = 0

for trainset, testset in kf.split(data):
# # # #     # train and test algorithm.
    start = time.time()
    algo.fit(trainset)
    print(time.time() - start)
    
    if algo_name == 'AgreeTrustAlgorithm':
        np.save(file_path_save_data+datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_epsilon_'+str(epsilon)+'_lambdak_'+str(lambdak)+'_trust_matrix_.npy', algo.algo.sim)
    elif algo_name == 'OdnovanAlgorithm':
        if algo.load:
            algo.algo.sim = np.load(file_path_save_data+datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_alpha_'+str(alpha)+'_trust_matrix_.npy')
        else:
            np.save(file_path_save_data+datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_alpha_'+str(alpha)+'_trust_matrix_.npy', algo.algo.sim)
    else:
        np.save(file_path_save_data+datasetname+str(kt)+'_'+algo_name+'_user_based_'+str(user_based)+'_sim_matrix_.npy', algo.sim)
    
   
    start = time.time()
    predictions = algo.test(testset)
    print(time.time() - start)

    # # # Compute and print RMSE and MAE
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
if algo_name == 'AgreeTrustAlgorithm':
    print(algo_name+'_predict_alog_'+str(predict_alog)+'_user_based_'+str(user_based)+'_epsilon_'+str(epsilon)+'_lambdak_'+str(lambdak))
elif algo_name == 'OdnovanAlgorithm':
    print(algo_name+'_predict_alog_'+str(predict_alog)+'_user_based_'+str(user_based)+'_alpha_'+str(alpha))
else:
    print(algo_name+'_user_based_'+str(user_based))
print('mean_rmse')
print(mean_rmse)
print('mean_mae')
print(mean_mae)

