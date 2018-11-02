from surprise import Dataset, evaluate
from surprise import Reader
from surprise import KNNBasic
from collections import defaultdict
import pandas as pd
import os, io
from surprise.accuracy import rmse
from surprise.accuracy import mae
import numpy as np
from surprise.model_selection import train_test_split

file_path = os.path.expanduser('~') + '/code/sci/recom/data/ml-100k/u1a.base'

reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
# trainset, testset = train_test_split(data, test_size=.2, train_size=None, random_state=100, shuffle=True)

# data = Dataset.load_builtin("ml-100k")
trainingSet = data.build_full_trainset()

trainset = trainingSet

testset = trainingSet.build_testset()

sim_options = {
    'name': 'cosine',
    'user_based': False
}


 
knn = KNNBasic(sim_options=sim_options)
knn.fit(trainset)


predictions = knn.test(testset)





temp_trust_matrix = np.zeros((trainset.n_users, trainset.n_items))



# df2 = pd.DataFrame(columns=['uid', 'iid', 'rui'])

df2= pd.DataFrame(columns=['uid', 'iid', 'rui'])

# df2 = pd.Series(['uid', 'uid', 'rui'])

    
for u,i,r in trainset.all_ratings():
    # temp_trust_matrix[u,i] = r
    # df2.append([u,i,r], ignore_index=True)
    df2 = df2.append({'uid': trainset.to_raw_uid(u), 'iid': trainset.to_raw_iid(i), 'rui':r}, ignore_index=True)
    # df2.append(pd.Series([u,i,r], index=['uid', 'iid', 'rui']),ignore_index=True)
# print(temp_trust_matrix)
print(df2)




np.savetxt("temp_trust_matrix.csv", temp_trust_matrix, delimiter=",")
# np.savetxt("testset.csv", testset, delimiter=",")
df = pd.DataFrame(predictions,columns=['uid', 'iid', 'rui', 'est', 'details'])

print(df)
# np.savetxt("df.csv", df, delimiter=",")

print(testset)
# print(testSet)
# def get_top3_recommendations(predictions, topN = 3):
     
#     top_recs = defaultdict(list)
#     for uid, iid, true_r, est, _ in predictions:
#         top_recs[uid].append((iid, est))
     
#     for uid, user_ratings in top_recs.items():
#         user_ratings.sort(key = lambda x: x[1], reverse = True)
#         top_recs[uid] = user_ratings[:topN]
     
#     return top_recs

# def read_item_names():
#     """Read the u.item file from MovieLens 100-k dataset and returns a
#     mapping to convert raw ids into movie names.
#     """
 
#     file_name = (os.path.expanduser('~') +
#                  '/.surprise_data/ml-100k/ml-100k/u.item')
#     rid_to_name = {}
#     with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
#         for line in f:
#             line = line.split('|')
#             rid_to_name[line[0]] = line[1]
 
#     return rid_to_name

rmse(predictions)
mae(predictions)