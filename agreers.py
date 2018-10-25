import pandas as pd
import math
from math import sqrt
import numpy as np
from math import ceil
import time
np.seterr(divide='ignore', invalid='ignore')

#   -- Scikit Learn --
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
##  Tool for calculating MSE
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
 
# UserID::MovieID::Rating::Timestamp
# UserID::Gender::Age::Occupation::Zip-code
# MovieID::Title::Genres

user_data = pd.read_table('data/ml-100k/u.data', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])

# user_data = pd.read_table('data/ml-1m/ratings.dat', sep='::', names=['user id', 'item id', 'rating', 'timestamp'])

# user_user = pd.read_table('data/ml-1m/users.dat', sep='::',names=['user id', 'age', 'gender', 'occupation', 'zip code'])
# movies_list = pd.read_table('data/ml-1m/movies.dat', sep='::',names=['movie id', 'tite','genre' ])

# user_data = user_data.head(1000)

n_users = user_data['user id'].unique().shape[0]
n_items = user_data['item id'].unique().shape[0]

#resequence user id's
user_data['item id'] = pd.factorize(user_data['item id'])[0] + 1



print("Number of users = " + str(n_users) + " | Number of movies = " + str(n_items))
#spliting data into train n test
train_data, test_data = train_test_split(user_data, random_state=4, train_size=.80, test_size=.20)

train_data_matrix = np.zeros((n_users, n_items))
test_data_matrix = np.zeros((n_users, n_items))

for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


rating_matrix = np.zeros((n_users,n_items))
for line in user_data.itertuples():
    rating_matrix[line[1]-1, line[2]-1] = line[3]


rating_matrix = rating_matrix.T
n_users = n_items

trust_matix = np.zeros((n_users,n_users))
def agreement(ratings, alpha):
    #for each unique user iterate
    for user_a in range(n_users):
        for user_b in range(user_a,n_users):
            if user_a != user_b:
                a_ratings = rating_matrix[user_a]
                b_ratings = rating_matrix[user_b]

                commonset = np.intersect1d(np.nonzero(rating_matrix[user_a]), np.nonzero(rating_matrix[user_b]))
                 
                common_set_length = len(commonset)

                trust = 0

                if(common_set_length > 0):
                    a_positive = a_ratings[commonset] > alpha
                    b_positive = b_ratings[commonset] > alpha

                    agreement = np.sum(np.logical_not(np.logical_xor(a_positive, b_positive)))

                    trust = agreement/common_set_length

                trust_matix[user_a,user_b] = trust



start = time.time()
agreement(rating_matrix, 2.5)
total = start - time.time()
print(total)
print(trust_matix)

# np.save('train_data_matrix.npy', train_data_matrix)
# np.save('test_data_matrix.npy', test_data_matrix)



np.save('trust_matix_it.npy', trust_matix)
t = np.load('trust_matix_it.npy')
plt.matshow(t);
plt.colorbar()
plt.show()

