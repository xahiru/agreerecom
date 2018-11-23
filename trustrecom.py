from __future__ import (absolute_import, division, print_function,             
                        unicode_literals)                                      
import pickle
import os

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time

from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset                                                     
from surprise import Reader                                                      
from surprise import dump
from surprise.accuracy import rmse
from surprise.accuracy import mae

from surprise import accuracy
from surprise.model_selection import train_test_split

from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import BaselineOnly

from sklearn.metrics.pairwise import pairwise_distances
import copy as cp
# import random
import matplotlib.pyplot as plt

class AgreeTrust(AlgoBase):
    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
    
    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, 

    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)
        return self

    def estimate(self, u, i):

        return self.the_mean