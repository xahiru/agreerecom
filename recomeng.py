from __future__ import (absolute_import, division, print_function,             
                        unicode_literals) 
import pandas as pd
import math
from math import sqrt
import numpy as np
from math import ceil
np.seterr(divide='ignore', invalid='ignore')

#    -- Scikit Learn --
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
##  Tool for calculating MSE
from sklearn.metrics import mean_squared_error

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


class RecomEng:

    # def __init__(self, data):
    #     self.data = data
    # #     self.rectype = rectype

    def predict(ratings, similarity, type='user'):
        # print(ratings.shape)
        # print(similarity.shape)
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            # ratings = ratings.T
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred[np.isnan(pred)] = 0
        pred[np.isinf(pred)] = 0
        return pred

    def gen_trust_matrix_leave_one_out(ratings, predict, ptype='user'):
        dim = 1
        
        #     # dim = 0
            # nums = ratings.T

        similarity = 1 - pairwise_distances(ratings, metric='cosine')
        prediction = predict(ratings, similarity, ptype)

        trust_matrix = np.zeros((ratings.shape[0],ratings.shape[0]))

        if(ptype == 'item'):
            trust_matrix = np.zeros((ratings.shape[1],ratings.shape[1]))

        # print('trust_matrix.shape')
        print(trust_matrix.shape)
        for x in range(ratings.shape[0]):
            ratings_new = ratings.copy()
            similarity_new = similarity.copy()
            
            ratings_new[x] = 0
            similarity_new[x,:] = 0
            similarity_new[:,x] = 0

            xhat_predict = predict(ratings_new, similarity_new,ptype)
            # print(np.any(np.isnan(xhat_predict)))

            predic_diff = abs(prediction - xhat_predict)
            predic_diff[np.isnan(predic_diff)] = 0
    
            # - (xhat_predict == 0).astype(bool).sum(dim) removes #of zero entries from numerator value
            numerator = (xhat_predict < 0.2).astype(bool).sum(dim) - (xhat_predict == 0).astype(bool).sum(dim)
            # print(numerator)

            denominator = xhat_predict.astype(bool).sum(dim)
            # print(denominator)
            trust_row = numerator/denominator
            trust_row[np.isnan(trust_row)] = 0
            trust_row[np.isinf(trust_row)] = 0
            # print(trust_row)
            
            trust_matrix[x] = trust_row

        return trust_matrix

    def pitsmarsh_trust(ratings, max_r, ptype='user'):
        if(ptype == 'item'):
            ratings = ratings.T
        
        trust_matrix = np.zeros((ratings.shape[0], ratings.shape[0]))
        # if(ptype == 'item'):
        #     trust_matrix = np.zeros((ratings.shape[1], ratings.shape[1]))
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
        print(trust_matrix.shape)
        return trust_matrix

    def agreement(ratings, alpha):
    #for each unique user iterate
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

                trust_matrix[user_a,user_b] = trust
    return trust_matrix

