from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from surprise import AlgoBase

from surprise import Dataset, evaluate
from surprise import Reader
from surprise import KNNWithMeans
from surprise import KNNWithMeansC
from collections import defaultdict
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.accuracy import rmse
from surprise.accuracy import mae
from surprise.model_selection import GridSearchCV
from surprise.agreements import agree_trust
from surprise.agreements import odonovan_trust_old
from surprise.model_selection import KFold
import os
import time
start=time.time()
print(start)

# reader = Reader(line_format='user item rating') #sep='\t',
# file_path = os.path.expanduser('~/.surprise_data/jester/jester_ratings.dat')
# data = Dataset.load_from_file(file_path, rating_scale=(-10, 10), reader=reader)

data = Dataset.load_builtin('jester')
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
# beta = 0

class MyOwnAlgorithm(AlgoBase):

    def __init__(self, k=40, min_k=1, alog=KNNWithMeans,user_based =True, beta=2.5, epsilon=0.9, sim_options={}, verbose=True, **kwargs):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.k = k
        self.min_k = min_k
        self.algo = alog(k=k,sim_options=sim_options,verbose=True)
        self.epsilon = epsilon
        self.beta = beta
        if user_based:
            self.ptype = 'user'
        else:
            self.ptype = 'item'



    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)
        # self.sim = self.compute_similarities(verbose=self.verbose)
        self.algo.fit(trainset)
        sim = self.algo.sim
        tsim,activityma = agree_trust(trainset, self.beta, self.epsilon, ptype=self.ptype, istrainset=True, activity=False)

        # Compute the average rating. We might as well use the
        # # trainset.global_mean attribute ;)
        # self.the_mean = np.mean([r for (_, _, r) in
        #                          self.trainset.all_ratings()])
        # mixsim = sim *tsim *tsim
        mixsim = (sim * tsim) +activityma
        self.algo.sim = mixsim

        return self

    def estimate(self, u, i):
    	# KNNWithMeans.test()
    	# self.algo.estimate(u,i)

        return self.algo.estimate(u,i)



class OdnovanAlgorithm(AlgoBase):
    def __init__(self, k=40, min_k=1, alog=KNNWithMeans,user_based =True, alpha=0.2, sim_options={}, verbose=True, **kwargs):
        self.algo = alog(k=k,sim_options=sim_options,verbose=True)
        self.alpha = alpha
        # self.testset = testset
        if user_based:
            self.ptype = 'user'
        else:
            self.ptype = 'item'
    def fit(self, trainset):
        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)
        self.algo.fit(trainset)
        print('OdnovanAlgorithm here')
        self.algo.sim = odonovan_trust_old(trainset, self.algo, ptype= self.ptype, alpha=self.alpha)

    def estimate(self, u, i):
        # KNNWithMeans.test()
        # self.algo.estimate(u,i)

        return self.algo.estimate(u,i)



####### GRID search for parameter optimization
# param_grid = {'k': [40],'epsilon':[0,0.01,0.1,0.5,0.9], 'user_based': [True], 'beta':[beta], 'sim_options': {'name': ['pearson'],
#                               # 'min_support': [1, 5],
#                               'user_based': [True]}}
# gs = GridSearchCV(MyOwnAlgorithm, param_grid, measures=['rmse', 'mae'], cv=2)

# gs.fit(data)

# # best RMSE score
# print(gs.best_score['rmse'])

# # combination of parameters that gave the best RMSE score
# print(gs.best_params['rmse'])


# #########
sim_options={'name':'pearson','user_based':True}
# cross_validate(MyOwnAlgorithm(k=40, alog=KNNWithMeans,user_based =True, beta=2.5, epsilon=0.9, sim_options=sim_options), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# cross_validate(KNNWithMeans(k=40,sim_options=sim_options), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# cross_validate(OdnovanAlgorithm(alog=KNNWithMeans(sim_options=sim_options), user_based=True, alpha=0.2), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
##
kf = KFold(n_splits=5)
algo = OdnovanAlgorithm(alog=KNNWithMeansC, sim_options=sim_options, user_based=True, alpha=0.2)

for trainset, testset in kf.split(data):
# #     # train and test algorithm.
    algo.fit(trainset)
#     predictions = algo.test(testset)

#     # Compute and print Root Mean Squared Error
#     accuracy.rmse(predictions, verbose=True)
