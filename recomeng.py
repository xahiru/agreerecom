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
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred[np.isnan(pred)] = 0
        pred[np.isinf(pred)] = 0
        return pred
    
    def gen_trust_matrix_leave_one_out(ratings,batch_size,prediction,type='user'):
        # if(type='item'):
        similarity = 1 - pairwise_distances(ratings, metric='cosine')
        trust_matrix = np.zeros((batch_size, batch_size))
        for x in range(batch_size):
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
            
            trust_matrix[x] = trust_row

        return trust_matrix