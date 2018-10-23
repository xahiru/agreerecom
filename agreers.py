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


print(rating_matrix.head())
trust_matix = np.zeros((n_users,n_users))
# def agreement(ratings):
    #for each unique user iterate

