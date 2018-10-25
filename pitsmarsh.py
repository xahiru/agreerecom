# https://pdfs.semanticscholar.org/d550/f9437c22d72be8fcadd3ad0fd77c66752a65.pdf
# Pitsilis and Marshall

# uncertainty u(a,b) between users a and b, which is computed as the average absolute
# difference of the ratings in the intersection of the two user’s profiles. The authors
# scale each difference by dividing it by the maximum possible rating, max(r):

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
##  Tool for calculating MSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt



train = np.load('train_data_matrix.npy')
test = np.load('test_data_matrix.npy')

max_r = 5

# user_similarity = pairwise_distances(train, metric='cosine')/max_r

# n_users = user_data['user id'].unique().shape[0]

# train = train[:210,]
# train = train.T

print(train)

trust_matrix = np.zeros((train.shape[0], train.shape[0]))

for a in range(train.shape[0]):
    for b in range(train.shape[0]):
        if (a!=b):
            r_a = train[a]
            r_b = train[b]

            common_index = np.intersect1d(np.nonzero(r_a),np.nonzero(r_b))
             
            normalized_dif = sum(abs(r_a[common_index] - r_b[common_index])/max_r)

            common = sum(common_index)

            score = 0
            if(common != 0):
                score = normalized_dif/common

            # print(score)
            trust_matrix[a,b] = score
            
            # print(a,b)

print(trust_matrix)
np.save('trust_matix_user_pitsmarsh.npy', trust_matrix)

plt.matshow(trust_matrix);
plt.colorbar()
plt.show()