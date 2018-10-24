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

user_data = pd.read_table('data/ml-1m/ratings.dat', sep='::', names=['user id', 'item id', 'rating', 'timestamp'])
# user_user = pd.read_table('data/ml-1m/users.dat', sep='::',names=['user id', 'age', 'gender', 'occupation', 'zip code'])
# movies_list = pd.read_table('data/ml-1m/movies.dat', sep='::',names=['movie id', 'tite','genre' ])

# user_data = user_data.head(300000)

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


# print(rating_matrix[0:10])
trust_matix = np.zeros((n_users,n_users))
def agreement(ratings, alpha):
    #for each unique user iterate
    for user_a in range(n_users):
        for user_b in range(user_a,n_users):
            if user_a != user_b:
                a_ratings = rating_matrix[user_a]
                b_ratings = rating_matrix[user_b]

                a_ix = np.nonzero(a_ratings)
                b_ix = np.nonzero(b_ratings)

                # commonset = np.intersect1d(np.nonzero(rating_matrix[user_a]), np.nonzero(rating_matrix[user_b]))
                commonset = np.intersect1d(a_ix, b_ix)

                    # print(commonset)
                    # print('a_ratings[commonset]')
                    # print(a_ratings[commonset])
                    # print('b_ratings[commonset]')
                    # print(b_ratings[commonset])

                    # print('len(commonset)')
                    
                common_set_length = len(commonset)
                # print(common_set_length)
                # print(user_a)
                



                #get #of common positive n common negatives


                trust = 0

                if(common_set_length > 0):
                    a_positive = a_ratings[commonset] > alpha
                    b_positive = b_ratings[commonset] > alpha

                        # 5,1,5,4,2,2 >2.4 = T,F,T,T,F,F
                        #                               =(po1)+(ne2) = (3) 
                        # 1,5,1,3,1,1 >2.4 = F,T,F,T,F,F   

 
                        # print("a_positive")
                        # print(a_positive)

                        # print("b_positive")
                        # print(b_positive)

                    agreed_positive = np.logical_and(a_positive, b_positive)
                        # print("agreed_positive")
                        # print(agreed_positive)

                    a_negative = a_ratings[commonset] < alpha
                    b_negative = b_ratings[commonset] < alpha

                        # print("a_negative")
                        # print(a_negative)

                        # print("b_negative")
                        # print(b_negative)

                    agreed_negative = np.logical_and(a_negative, b_negative)
                        # print("agreed_negative")
                        # print(agreed_negative)

                    agreement = np.sum(agreed_positive) + np.sum(agreed_negative)

                        # print('agreement')
                        # print(agreement)
                        
                    trust = agreement/common_set_length

                trust_matix[user_a,user_b] = trust

                # trust = total agreement/total ratings
                # total agreement = agreed (positives + negatives)
                # a_common = [1,2,3,5,1,5] positive = [F,F,T,T,F,T] , (x>2.5)? T : F
                # b_common = [3,4,2,4,1,3] positive = [T,T,F,T,F,T] , (x>2.5)? T : F
                # total agreement = 1+1
                # trust = (1+1)/6



                #normalization userser range = highest-lowest OR score- mean.
                # [5,3,3,5,3,5,4,5] [2,0,0,2,0,2,1,0]  
                # [5,1,1,5,1,5,4,5] [4,0,0,4,0,4,3,4]



start = time.time()
agreement(rating_matrix, 2.5)
total = start - time.time()
print(total)
print(trust_matix)

# plt.imshow(trust_matix);
# plt.colorbar()
# plt.show()

plt.matshow(trust_matix);
plt.colorbar()
plt.show()

# np.save('trust_matix.npy', trust_matix)