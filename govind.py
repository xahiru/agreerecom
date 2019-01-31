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

######################################### running parameters #############################
max_r = 5
beta = max_r/2
alpha=0.2
epsilon=0.1
# epsilon=0.9 gives MAE 0.6988

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

if datasetname == 'jester':
	beta = 0

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
	# sim2 = cp.deepcopy(sim)

	# tsim,activityma = agree_trust(trainset, beta, epsilon, ptype=ptype, istrainset=True, activity=False)
	# mixsim = sim *tsim *tsim
	# algo.sim = mixsim
	tr, comon, noncom = agree_trust_op(trainset, beta, epsilon, algo.sim, ptype=ptype, istrainset=True, activity=False)
	algo.sim = (sim * tr * tr) + (noncom)
	# algo.sim = (sim * tr) + (noncom *sim2)

predictions=algo.test(testset)
print(datasetname)
if base_line != True:
	print('((sim * tr) + (noncom *sim2))')
	# print('(sim * tsim) +activityma')
else:
	print('base_line')
print(ptype)
print(k)
print('epsilon')
print(epsilon)
rmse(predictions)
mae(predictions)

# fig , ax = plt.subplots()
# ax = fig.add_subplot(111)
# ax.plot(sim)
# plt.matshow(algo.sim);
# plt.colorbar()
# fig.show()

# print(algo.counts)
# plt.plot(algo.counts)
# plt.ylabel('algo.counts')
# plt.show()

# print(activityma[30,40])
# print(activityma[40,30])

# print(activityma[30,54])
# print(activityma[54,30])

print(noncom[30,40])
print(noncom[40,30])

print(noncom[30,54])
print(noncom[54,30])

plt.matshow(comon);
plt.colorbar()
# plt.show()
plt.savefig(datasetname+str(user_based)+'comon.png')

plt.matshow(tsim);
plt.colorbar()
# plt.show()
plt.savefig(datasetname+str(user_based)+'tsim.png')


plt.matshow(noncom);
plt.colorbar()
# plt.show()
plt.savefig(datasetname+str(user_based)+'noncom.png')



# fig2 = plt.figure(2)
# plt.matshow(mixsim);
# plt.colorbar()
# plt.show()
