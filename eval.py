import pandas as pd
import math
from math import sqrt
import numpy as np
from math import ceil
np.seterr(divide='ignore', invalid='ignore')
from numpy import *


#    -- Scikit Learn --
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
##  Tool for calculating MSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# import matplotlib.pyplot as plt

from recomeng import RecomEng as rec



trust_matix = np.load('data/ml-100k/odn/trust_matix_user_test.npy')
trust_matix_it = np.load('data/ml-100k/odn/trust_matix_item_test.npy')
# trust_matix = np.load('data/ml-100k/agree/trust_matix_useralpha_beta2.5.npy')

# trust_matix = np.load('data/ml-100k/agree/trust_matix_user.npy')
# trust_matix_it = np.load('data/ml-100k/agree/trust_matix_item.npy')


# trust_matix = np.load('data/ml-100k/odn/trust_matix_user.npy')
# trust_matix_it = np.load('data/ml-100k/odn/trust_matix_item.npy')

# trust_matix = np.load('data/ml-100k/pits/trust_matix_user.npy')
# trust_matix_it = np.load('data/ml-100k/pits/trust_matix_item.npy')


train = np.load('train_data_matrix3.npy')
# train = np.array([[5, 4, 3, 3, 5],[0, 0, 0, 0, 0],[4, 4, 3, 3, 5],[4, 4, 3, 1, 3],[3, 3, 2, 2, 5],[2, 0, 2, 1, 5],[2, 4, 5, 4, 1],[2, 2, 1, 1, 1],[1, 1, 1, 2, 2],[0, 0, 0, 0, 0]])


test = np.load('test_data_matrix_3.npy')
# test =  np.array([[0, 0, 0, 0, 0],[2, 3, 2, 1, 2],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[1, 2, 3, 4, 3]])

# print('trust_matix.shape')
# print(trust_matix.shape)

# print('trust_matix_it.shape')
# print(trust_matix_it.shape)




def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    # prediction[prediction >= 1E308] = 0
    # ground_truth[ground_truth >= 1E308] = 0
    return sqrt(mean_squared_error(prediction, ground_truth))

def ae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sum(abs(prediction-ground_truth))/sum(abs(ground_truth))

def mae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction,ground_truth)


print('cosine_user_similarity calculation ... user')

cosine_user_similarity = 1 - pairwise_distances(train, metric='cosine')

print('cosine_user_similarity calculation ... item')

cosine_item_similarity = 1 - pairwise_distances(train.T, metric='cosine')

# print('cosine_user_similarity.shape')
# print(cosine_user_similarity.shape)

# print('cosine_item_similarity.shape')
# print(cosine_item_similarity.shape)

print('similarity calculation done')
# print(train.shape)

# user - user CF:
user_prediction = rec.predict(train, cosine_user_similarity, type='user')
# item - item CF:
item_prediction = rec.predict(train, cosine_item_similarity, type='item')

print('user_prediction item_prediction calculation done')
# print('item_prediction.shape')
# print(item_prediction.shape)

# print('User-based CF RMSE: ' + str(rmse(user_prediction, test)) + '|' + ' User-based CF AE: ' + str(ae(user_prediction, test)) + '|' + ' User-based CF MAE: ' + str(mae(user_prediction, test)))
# print('Item-based CF RMSE: ' + str(rmse(item_prediction, test)) + '|' + ' Item-based CF AE: ' + str(ae(item_prediction, test)) + '|' + ' Item-based CF MAE: ' + str(mae(item_prediction, test)))

print('User-based CF RMSE: ' + str(rmse(user_prediction, test)) + '|' + ' User-based CF MAE: ' + str(mae(user_prediction, test)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test)) + '|' + ' Item-based CF MAE: ' + str(mae(item_prediction, test)))



# user - user TCF:
print('user TCF:')
# print('trust_matix.shape')
# print(trust_matix.shape)
tcf_user_prediction = rec.predict(train, trust_matix, type='user')
# print('tcf_user_prediction.shape')
# print(tcf_user_prediction.shape)

# item - item TCF:
print('item TCF:')
tcf_item_prediction = rec.predict(train, trust_matix_it, type='item')
# print('tcf_item_prediction.shape')
# print(tcf_item_prediction.shape)

tcf_user_prediction[isnan(tcf_user_prediction)] =0

tcf_item_prediction[isnan(tcf_item_prediction)] = 0

# print("pd.isnull(tcf_item_prediction).sum() > 0")

# print(pd.isnull(tcf_item_prediction).sum() > 0)


# print(tcf_item_prediction)

print('User-based TCF RMSE: ' + str(rmse(tcf_user_prediction, test)) + '|' + ' User-based CF MAE: ' + str(mae(tcf_user_prediction, test)))
print('Item-based TCF RMSE: ' + str(rmse(tcf_item_prediction, test)) + '|' + ' Item-based CF MAE: ' + str(mae(tcf_item_prediction, test)))



# combined_sim_trust_user = (cosine_user_similarity + trust_matix)/2
combined_sim_trust_user = (2*(trust_matix*cosine_user_similarity))/(trust_matix + cosine_user_similarity)
# a = combined_sim_trust_user.flatten()
# print('a')
# print(len(a))
# hmean_sim = len(a) / np.sum(1.0/a) 
# print('hmean_sim')
# print(hmean_sim)
# hmean_sim_new = np.reshape(hmean_sim,(int(len(a)/943), 943))
# user - user CF:
print('combined_user_prediction  - user CF:')
combined_user_prediction = rec.predict(train, combined_sim_trust_user, type='user')
# combined_user_prediction = rec.predict(train, hmean_sim, type='user')


combined_user_prediction[isnan(combined_user_prediction)] =0


# combined_sim_trust_item = (cosine_item_similarity + trust_matix_it)/2
combined_sim_trust_item = (2*(trust_matix_it*cosine_item_similarity))/(trust_matix_it + cosine_item_similarity)

print('combined_user_prediction  - item CF:')
combined_item_prediction = rec.predict(train, combined_sim_trust_item, type='item')
combined_item_prediction[isnan(combined_item_prediction)] =0

print('User-based TCF_Sim RMSE: ' + str(rmse(combined_user_prediction, test)) + '|' + ' User-based CF MAE: ' + str(mae(combined_user_prediction, test)))
print('Item-based TCF_Sim RMSE: ' + str(rmse(combined_item_prediction, test)) + '|' + ' Item-based CF MAE: ' + str(mae(combined_item_prediction, test)))




# plt.matshow(trust_matrix);
# plt.colorbar()
# plt.show()