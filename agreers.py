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

from recomeng import RecomEng as rec
 
# UserID::MovieID::Rating::Timestamp
user_data = pd.read_table('data/ml-100k/u.data', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])

n_users = user_data['user id'].unique().shape[0]
n_items = user_data['item id'].unique().shape[0]


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


start = time.time()


ptype_list = ['user','item']
alog_list = ['agree','odn', 'pits']

alpha = 2.5
beta = 0.2
alpha_beta = 0
max_r = 5

#For test sampling
# n_users = 10
# n_items = 100
# train_data_matrix = train_data_matrix[:n_users,:n_items]
# test_data_matrix = test_data_matrix[:n_users,:n_items]


for alog in alog_list:
    for ptype in ptype_list:
        if ptype == 'user':
            trust_matrix = np.zeros((n_users,n_users))
            # train_data_matrix = train_data_matrix_orginal
        else:
            rating_matrix = rating_matrix.T
            train_data_matrix = train_data_matrix.T
            test_data_matrix = test_data_matrix.T
            n_users = n_items
            trust_matrix = np.zeros((n_items,n_items))

        if alog == 'agree':
            trust_matrix = rec.agreement(test_data_matrix,alpha)
            alpha_beta = alpha
        elif alog == 'odn':
            trust_matrix = rec.gen_trust_matrix_leave_one_out(train_data_matrix, rec.predict,test_data_matrix, ptype)
            # trust_matrix = rec.gen_trust_matrix_leave_one_out(train_data_matrix, rec.predict, '')
        else:
            trust_matrix = rec.pitsmarsh_trust(train_data_matrix, max_r, ptype)
            alpha_beta = max_r
        

        np.save('data/ml-100k/'+str(alog)+'/trust_matix_'+str(ptype)+'_test.npy', trust_matrix)
        # print('loaded '+ptype+' matrix size' + str(np.load('data/ml-100k/'+str(alog)+'/trust_matix_'+str(ptype)+'_test.npy').shape))


# save train test
# np.save('data/ml-100k/train_data_matrix.npy',train_data_matrix)
# np.save('data/ml-100k/test_data_matrix.npy',test_data_matrix)

total = start - time.time()
print(total)

# t = np.load('data/ml-100k/'+str(alog)+'/trust_matix_'+str(ptype)+'alpha_beta'+str(alpha_beta)+'.npy')
# print(trust_matrix)
# plt.matshow(t);
# plt.colorbar()
# plt.show()