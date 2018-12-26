from surprise import Dataset
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import surprise; print("surprise", surprise.__version__)

data = Dataset.load_builtin('jester')
trainset, testset = train_test_split(data, test_size=.2, train_size=None, random_state=100, shuffle=True)
sim_options={'name':'pearson','user_based':True}
algo = KNNWithMeans(sim_options=sim_options,verbose=True)
algo.fit(trainset)
predictions=algo.test(testset)
rmse(predictions)
