import pandas as pd
import math
from math import sqrt
import numpy as np
from math import ceil
np.seterr(divide='ignore', invalid='ignore')

#   -- Scikit Learn --
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
##  Tool for calculating MSE
from sklearn.metrics import mean_squared_error
 
# UserID::MovieID::Rating::Timestamp
# UserID::Gender::Age::Occupation::Zip-code
# MovieID::Title::Genres

user_data = pd.read_table('data/ml-1m/ratings.dat', sep='::', names=['user id', 'item id', 'rating', 'timestamp'])
# user_user = pd.read_table('data/ml-1m/users.dat', sep='::',names=['user id', 'age', 'gender', 'occupation', 'zip code'])
# movies_list = pd.read_table('data/ml-1m/movies.dat', sep='::',names=['movie id', 'tite','genre' ])

user_data = user_data.head(100)

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


print(rating_matrix[0:10])
trust_matix = np.zeros((n_users,n_users))
def agreement(ratings, alpha):
    #for each unique user iterate
    for user_a in range(n_users):
        for user_b in range(n_users):
            if user_a != user_b:
                a_ratings = rating_matrix[user_a]
                b_ratings = rating_matrix[user_b]

                a_ix = np.nonzero(a_ratings)
                b_ix = np.nonzero(b_ratings)

                #get the non zero values
                # np.nonzero(a > 3)
                # commonset = set(a_ix).intersection(b_ix)
                commonset = np.intersect1d(a_ix, b_ix)

                if user_a <10 and user_b <10:
                    print(commonset)
                    print('a_ratings[commonset]')
                    print(a_ratings[commonset])
                    print('b_ratings[commonset]')
                    print(b_ratings[commonset])
                



                #get #of common positive n common negatives

                    a_positive = a_ratings[commonset] > alpha
                    b_positive = b_ratings[commonset] > alpha

                    agreed_positive = np.logical_and(a_positive, b_positive)

                    # print(agreed_positive)

                    a_negative = a_ratings[commonset] < alpha
                    b_negative = b_ratings[commonset] < alpha

                    agreed_negative = np.logical_and(a_positive, b_positive)
                    agreement = np.sum(agreed_positive) + np.sum(agreed_negative)

                    trust = agreement/len(commonset)
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




agreement(rating_matrix, 2.5)

print(trust_matix)