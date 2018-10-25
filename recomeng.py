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

class RecomEng:

    # def __init__(self, rectype, t_matrix):
    #     self.trust_matrix = t_matrix
    #     self.rectype = rectype

    def predict(ratings, similarity, type='user'):
        # print(ratings)
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            #You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            # print(ratings.shape)
            # print(similarity.shape)
            # print(np.array([np.abs(similarity).sum(axis=1)]).shape)
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])   
        return pred

    # def trust_predict(ratings, trust_weights, type='user'):
    #     if type == 'user':
    #         mean_user_rating = ratings.mean(axis=1)
    #         ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    #         # print(trust_weights.shape)
    #         # print(ratings_diff.shape)
    #         pred = mean_user_rating[:, np.newaxis] + trust_weights.dot(ratings_diff) / np.array([np.abs(trust_weights).sum(axis=1)]).T
    #     elif type == 'item':
    #         # print(ratings.shape)
    #         # print(trust_weights.shape)
    #         # print(np.array([np.abs(trust_weights).sum(axis=1)]).T.shape)
    #         pred = ratings.dot(trust_weights) / np.array([np.abs(trust_weights).sum(axis=1)])
    #         # pred = trust_weights.dot(ratings) / np.array([np.abs(trust_weights).sum(axis=1)]).T
    #     return pred