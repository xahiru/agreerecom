from surprise import Dataset
from surprise import KNNWithMeans
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from surprise.accuracy import mae
from surprise.agreements import agree_trust
from surprise.agreements import agree_trust_old
import matplotlib.pyplot as plt
import os
from surprise import Reader
import numpy as np

######################################### running parameters #############################
max_r = 5
beta = max_r/2
alpha=0.2
epsilon=0.9

# file_path = os.path.expanduser('~') + '/Code/paper/agree/agreerecom/data/ml-100k/u1b.base'
#     # reader = Reader(line_format='user item rating timestamp', sep=',')
# reader = Reader(line_format='user item rating timestamp', sep='	', rating_scale=(1, 5), skip_lines=1)
# data = Dataset.load_from_file(file_path, reader=reader)

# datasetname = 'ml-latest-small'
datasetname = 'ml-20m'
# datasetname = 'jester'
data = Dataset.load_builtin(datasetname)
# data = Dataset.load_builtin('jester')
user_based = True
base_line = False
k= 40

if user_based:
	ptype = 'user'
else:
	ptype = 'item'


trainset, testset = train_test_split(data, test_size=.2, train_size=None, random_state=100, shuffle=True)
sim_options={'name':'pearson','user_based':user_based}
algo = KNNWithMeans(k=k,sim_options=sim_options,verbose=True)
# algo = SVD()
algo.fit(trainset)

if base_line != True:
	sim = algo.sim
	tsim,activityma = agree_trust(trainset, beta, epsilon, ptype=ptype, istrainset=True, activity=False)
	# mixsim = (sim *tsim) / 2
	mixsim = sim *tsim *tsim
	# mixsim *= activityma
	# # sim[min_index] = activityma[min_index]
	algo.sim = mixsim

predictions=algo.test(testset)
print(datasetname)
if base_line != True:
	print('sim *tsim *tsim')
else:
	print('base_line')
print(ptype)
print(k)
rmse(predictions)
mae(predictions)

# fig1 = plt.figure(1)
# plt.matshow(algo.sim);
# plt.colorbar()
# plt.show()

# plt.matshow(agreenormal_trust);
# plt.colorbar()
# plt.show()

# fig2 = plt.figure(2)
# plt.matshow(mixsim);
# plt.colorbar()
# plt.show()
