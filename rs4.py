import pandas as pd
import math
from math import sqrt
import numpy as np
from math import ceil
np.seterr(divide='ignore', invalid='ignore')


#https://github.com/pranzell/Recommender-Systems/blob/master/Recommendation%20Problem%20-%20Movie%20Data%20Set.ipynb


#    -- Scikit Learn --
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
user_user = pd.read_table('data/ml-1m/users.dat', sep='::',names=['user id', 'age', 'gender', 'occupation', 'zip code'])
movies_list = pd.read_table('data/ml-1m/movies.dat', sep='::',names=['movie id', 'tite','genre' ])

n_users = user_data['user id'].unique().shape[0]
n_items = user_data['item id'].unique().shape[0]
# user_data2 = user_data.copy()
user_data['item id'] = pd.factorize(user_data['item id'])[0] + 1

# print("user_data['user id'].unique()")
# print(user_data['user id'].unique())


# print("user_data['item id'].unique()")
# print(user_data['item id'].unique())

# print("n_users")
# print(n_users)

# print("n_items")
# print(n_items)
# print(user_user.head())

# print(movies_list.head())

# print(movies_list.head()['tite'])
# user_batch_size = n_users
# item_batch_size = n_items

print("Number of users = " + str(n_users) + " | Number of movies = " + str(n_items))

train_data, test_data = train_test_split(user_data, random_state=4, train_size=.80, test_size=.20)



# (Training) User x Item Matrix --
train_data_matrix = np.zeros((n_users, n_items))
# print("train_data_matrix.shape")
# print(train_data_matrix.shape)


# print("train_data.shape")
# print(train_data.shape)

# print("train_data.head()")
# print(train_data.head())

for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    # print(line)

# (Testing) User x Item Matrix --
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

print(np.count_nonzero(train_data_matrix==0))
    # user - user similarity Matrix (943x943) :
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

# item -item similarity Matrix (1682x1682) :
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

# user - user similarity Matrix (943x943) :
cosine_user_similarity = 1 - user_similarity

# item -item similarity Matrix (1682x1682) :
cosine_item_similarity = 1 - item_similarity


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


def trust_predict(ratings, trust_weights, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        # print(trust_weights.shape)
        # print(ratings_diff.shape)
        pred = mean_user_rating[:, np.newaxis] + trust_weights.dot(ratings_diff) / np.array([np.abs(trust_weights).sum(axis=1)]).T
    elif type == 'item':
        # print(ratings.shape)
        # print(trust_weights.shape)
        # print(np.array([np.abs(trust_weights).sum(axis=1)]).T.shape)
        pred = ratings.dot(trust_weights) / np.array([np.abs(trust_weights).sum(axis=1)])
        # pred = trust_weights.dot(ratings) / np.array([np.abs(trust_weights).sum(axis=1)]).T
    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def gen_trust_matrix_leave_one_out(ratings,similarity,batch_size,prediction, ptype):
    trust_matrix = np.zeros((batch_size, batch_size))
    dim = 1
    profiles_sep_option = 'no'
    # percentage = 80 #80 20    
    for x in range(batch_size):
        ratings_new = ratings.copy()
        similarity_new = similarity.copy()
        
        if ptype == 'item':
            ratings_new = ratings_new.T
            dim = 0

        ratings_new[x] = 0
        similarity_new[x,:] = 0
        similarity_new[:,x] = 0
        # print(similarity_new)

        # ratings_new_matrix2 = ratings_new[:]
        # np.delete(ratings_new_matrix2, x, 0)
        # print(ratings_new[x])

        if ptype == 'item':
            ratings_new = ratings_new.T

        
        # print(ratings_new.shape)
        # print(similarity.shape)

        xhat_predict = predict(ratings_new, similarity_new,ptype)

        # print(xhat_predict)
        # print(prediction)

        xhat_predict[np.isnan(xhat_predict)] = 0
        # print(np.any(np.isnan(xhat_predict)))

        predic_diff = abs(prediction - xhat_predict)

        xhat_predict[np.isnan(xhat_predict)] = 0
        predic_diff[np.isnan(predic_diff)] = 0
        
        # print(predic_diff)
        # # print(predic_diff[predic_diff < 2])
        # # print(predic_diff.shape)
        # # print(xhat_predict.shape)
        
        # xc = xhat_predict.copy()
        # idx_xc = xc == 0
        

        # pd_diff = predic_diff.copy()
        # pd_diff[idx_xc] = 0

        # print(pd_diff)

        # b = predic_diff.copy()
        # a = np.array(b)
        # c = np.where(a < kk)
        # b[c] = 1
        # b.reshape(predic_diff.shape)
        # # print(b)
        # print(b.astype(bool).sum(dim))
        # print('kkk')

        numerator = (xhat_predict < 0.2).astype(bool).sum(dim) - (xhat_predict == 0).astype(bool).sum(dim)

        # print(numerator)
        # print((xhat_predict == 0).astype(bool).sum(dim))
        denominator = xhat_predict.astype(bool).sum(dim)
        # print(denominator)
        # print(pd_diff.astype(bool).sum(dim))

        # total_predictions = (xhat_predict != 0).sum(dim)

        # # total_predictions = (xhat_predict != 0).astype(bool).sum(dim)


        # total_predictions[np.isnan(total_predictions)] = 0
        
        # print(total_predictions)
        # np.where(np.isnan(predic_diff), predic_diff, 0)
        
        # trust_row = ((predic_diff <0.1).sum(dim))/predic_diff.shape[dim]
        # trust_row = ((predic_diff <1.8).sum(dim))/predic_diff.sum(dim)
        trust_row = numerator/denominator
        trust_row[np.isnan(trust_row)] = 0
        trust_row[np.isinf(trust_row)] = 0
        # np.where(np.isinf(trust_row), trust_row, 0)
        # # np.where(np.isnan(trust_row), trust_row, 0)
        # if ((x/batch_size)*100) >= 80:
        #     # print('append zeros to consumers')
        #     fill_length = ceil(len(trust_row)*0.8)

        #     # trust_row[fill_length] = 0
        # print(trust_row)
        
        trust_matrix[x] = trust_row

    # trust_matrix[np.isnan(trust_matrix)] = 0
    # trust_matrix[np.isinf(trust_matrix)] = 0
    return trust_matrix

def get_harmonic_mean(trust_matrix, cos_similarity):
    return (2*(trust_matrix*cos_similarity))/(trust_matrix + cos_similarity)

def get_trust_prediction(train_data_matrix,cos_similarity,batch_size, prediction, ptype):
    trust_matrix = gen_trust_matrix_leave_one_out(train_data_matrix, cos_similarity,batch_size, prediction, ptype)
    trust_weights = get_harmonic_mean(trust_matrix,cos_similarity)
    
    return predict(train_data_matrix,trust_weights,ptype)


def get_trust_filtered_prediction(train_data_matrix,cos_similarity,batch_size, prediction, ptype):
    trust_matrix = gen_trust_matrix_leave_one_out(train_data_matrix, cos_similarity,batch_size, prediction, ptype)
    
    # sim_filter = trust_matrix > 1.5
    # print(trust_matrix)
    # print(cos_similarity)
    # cos_similarity = cos_similarity[sim_filter]
    # print(cos_similarity)
    # return predict(train_data_matrix,cos_similarity,ptype)



def eval_single(train_data_matrix,cosine_user_similarity,cosine_item_similarity,user_batch_size, item_batch_size,user_prediction, item_prediction, test_data_matrix):
    tw_user_predictions = get_trust_prediction(train_data_matrix[:user_batch_size,:], cosine_user_similarity[:user_batch_size,:user_batch_size],user_batch_size,user_prediction[:user_batch_size,:],ptype='user')
    tw_item_predictions = get_trust_prediction(train_data_matrix[:,:item_batch_size], cosine_item_similarity[:item_batch_size,:item_batch_size],item_batch_size,item_prediction[:,:item_batch_size], ptype='item')

    tw_item_predictions[np.isnan(tw_item_predictions)] = 0
    tw_item_predictions[np.isinf(tw_item_predictions)] = 0

    tw_user_predictions[np.isnan(tw_user_predictions)] = 0
    tw_user_predictions[np.isinf(tw_user_predictions)] = 0

    print('Tw-based user CF RMSE: ' + str(rmse(tw_user_predictions, test_data_matrix[:user_batch_size,:])))
    print('Tw-Items-based user CF RMSE: ' + str(rmse(tw_item_predictions, test_data_matrix[:,:item_batch_size])))




def proccess_batch(batch_size,testp,prdict,cosim,ptype):

    if ptype == 'item':
        batch_count = int(n_items/batch_size)
        ini = 0
        for x in range(batch_count):
            t = train_data_matrix[:,ini:ini+batch_size]
            # print(t)
            sim = cosim[ini:ini+batch_size,ini:ini+batch_size]
            p = prdict[:,ini:ini+batch_size]
            test = testp[:,ini:ini+batch_size]

            tp = get_trust_prediction(t,sim,batch_size,p,ptype)
            # print(np.any(np.isnan(tp)))
            # print(str(rmse(tp, test)))
            # print(tp)
            # test[:item_batch_size,:]

            # print('similarity between '+ str(ini) +', '+ str(ini+item_batch_size))
            tp[np.isnan(tp)] = 0
            # print(sim)
            # print(test[:,ini:ini+batch_size])
            print(str(rmse(tp, test[:,ini:ini+batch_size])))
            ini += item_batch_size
    else:
        batch_count = int(n_users/batch_size)
        ini = 0
        for x in range(batch_count):
            
            t = train_data_matrix[ini:ini+batch_size,:]

            sim = cosim[ini:ini+batch_size,ini:ini+batch_size]
            p = prdict[ini:ini+batch_size,:]
            tp = get_trust_prediction(t,sim,batch_size,p,ptype)
            test = testp[ini:ini+batch_size,:]
            print(str(rmse(tp, test[ini:ini+batch_size,:])))
            ini += batch_size

user_batch_size = 300#n_users
item_batch_size = 300#n_items


# user - user CF:
user_prediction = predict(train_data_matrix, cosine_user_similarity, type='user')
# item - item CF:
item_prediction = predict(train_data_matrix, cosine_item_similarity, type='item')

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

# ptype = 'user'
# proccess_batch(10, train_data_matrix, user_prediction, cosine_user_similarity,  ptype)

ptype = 'item'
# proccess_batch(item_batch_size, train_data_matrix, item_prediction, cosine_item_similarity,  ptype)


eval_single(train_data_matrix,cosine_user_similarity,cosine_item_similarity,user_batch_size, item_batch_size,train_data_matrix, train_data_matrix, test_data_matrix)

# get_trust_filtered_prediction(train_data_matrix[:,:item_batch_size], cosine_item_similarity[:item_batch_size,:item_batch_size],item_batch_size,item_prediction[:,:item_batch_size], ptype='item')
# get_trust_filtered_prediction(train_data_matrix[:user_batch_size,:], cosine_item_similarity[:user_batch_size,:user_batch_size],user_batch_size,user_prediction[:user_batch_size,:], ptype='user')


