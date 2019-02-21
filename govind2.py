from surprise import Dataset
from surprise import KNNWithMeans
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from surprise.accuracy import mae
from surprise.agreements import agree_trust
from surprise.agreements import agree_trust_old
from surprise.agreements import agree_trust_op
import matplotlib.pyplot as plt
import os
from surprise import Reader
import numpy as np
import copy as cp
import pandas as pd    


######################################### running parameters #############################
# file_path = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u1b.base'
# with open(file_path,'r') as f:
#     # next(f) # skip first row
#     df = pd.DataFrame(l.rstrip().split() for l in f)
# file_path = os.path.expanduser('~') + '/.surprise_data/ml-latest-small/ratings.csv'
# df = pd.read_csv(file_path) 

# print(df)

#     # reader = Reader(line_format='user item rating timestamp', sep=',')
# reader = Reader(line_format='user item rating timestamp', sep='	', rating_scale=(1, 5), skip_lines=1)
# data = Dataset.load_from_file(file_path, reader=reader)

# # datasetname = 'ml-latest-small'
# datasetname = 'ml-20m'
# # datasetname = 'jester'



# fig2 = plt.figure(2)
# plt.matshow(mixsim);
# plt.colorbar()
# plt.show()
