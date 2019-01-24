"""
The :mod:`similarities <surprise.similarities>` module includes tools to
compute similarity metrics between users or items. You may need to refer to the
:ref:`notation_standards` page. See also the
:ref:`similarity_measures_configuration` section of the User Guide.

Available similarity measures:

.. autosummary::
    :nosignatures:

    cosine
    msd
    pearson
    pearson_baseline
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np
# from cython.parallel import prange
# from cython.view cimport array as cvarray
# import multiprocessing

import time
import copy as cp
import pandas as pd


from six.moves import range
from six import iteritems


# @cython.wraparound(False)
# @cython.cdivision(True)
# @cython.nonecheck(False)
def agree_trust(trainset, beta, epsilon, ptype='user', istrainset=True, activity=False):
    print('======================== agree_trust |START|========================')
    start = time.time()
    cdef np.ndarray[np.double_t, ndim=2] trust_matrix, activity_matrix
    cdef int user_a, user_b, common_set_length, i, j, lenA, lenB, a_positive, b_positive, agreement, a_count, b_count, shorts_length
    cdef np.ndarray[np.double_t, ndim=1] a_ratings, b_ratings
    cdef np.ndarray[np.double_t, ndim=2] ratings
    cdef double a_val, b_val, trust, activity_val
    # cdef int[::1] intersect

    
    if istrainset == True:
        ratings = np.zeros((trainset.n_users, trainset.n_items))
        for u,i,r in trainset.all_ratings():
            ratings[u,i] =r
    else:
        ratings = trainset

    if ptype=='item':
        ratings = ratings.T

    n_x = ratings.shape[0]
    trust_matrix = np.zeros((n_x, n_x), np.double)
    activity_matrix = np.zeros((n_x, n_x), np.double)

    
    for user_a in range(n_x):
        trust_matrix[user_a, user_a] = 1
        for user_b in range(n_x):
        # for user_b in range(user_a + 1, n_x):
            a_ratings = ratings[user_a]
            b_ratings = ratings[user_b]
            # print('a_ratings')
            # print(a_ratings)
            # print('np.nonzero(a_ratings)')
            # print(np.nonzero(a_ratings))

            # a_ratings = a_ratings[np.nonzero(a_ratings)]
            # b_ratings = b_ratings[np.nonzero(b_ratings)]
            
            i = 0
            # j = 0
            common_set_length = 0
            # a_positive = 0
            # b_positive = 0
            lenA = a_ratings.shape[0]
            lenB = b_ratings.shape[0]
            a_count = 0
            b_count = 0

            agreement = 0

            # print('lenA')
            # print(lenA)
            # print('lenB')
            # print(lenB)
            # if lenA > lenB:
            #     shorts_length = lenB
            # else:
            #     shorts_length = lenA

            while (i < lenA):
                a_val = a_ratings[i]
                b_val = b_ratings[i]
                # print(a_val)
                # print(b_val)
                if a_val != 0 and b_val != 0:
                    common_set_length += 1
                    if a_val > beta and b_val > beta:
                        agreement += 1
                    elif a_val < beta and b_val < beta:
                        agreement += 1
                    elif a_val == beta and b_val == beta:#in ml-100k this will never be true for beta 2.5 since ratings are integers
                        agreement += 1
                elif a_val != 0:
                    a_count += 1
                elif b_val != 0:
                    b_count += 1
                i += 1

            # while (i < shorts_length):
            #     a_val = a_ratings[i]
            #     b_val = b_ratings[i]
            #     common_set_length += 1
            #     if a_val > beta and b_val > beta:
            #         agreement += 1
            #     elif a_val < beta and b_val < beta:
            #         agreement += 1
            #     elif a_val == beta and b_val == beta:#in ml-100k this will never be true for beta 2.5 since ratings are integers
            #         agreement += 1
            #     # else a_val != 0:
            #     #     a_count += 1
            #     # else b_val != 0:
            #     #     b_count += 1
            #     i += 1

            trust = 0
            activity_val = 0
            trust = agreement/(common_set_length+epsilon)
            # trust = np.sqrt(trust)
            # print(trust)
            # activity_val = a_count/(b_count+epsilon)
            activity_val = 1/(1+np.exp(-np.abs((a_count - b_count)/(a_count+b_count+epsilon))))
            # activity_val = 1/(1+np.exp(-(a_count - b_count)/(a_count+b_count+epsilon)))
            # activity_val = 1-1/(1+np.exp(-np.abs(a_count - b_count)))
            # activity_val = 1/(activity_val+0.9)
            activity_matrix[user_a,user_b] = activity_val
            trust_matrix[user_a,user_b] = trust
        activity_matrix[user_a, user_a] = 1
    # print('======================== agree_trust |END|========================')
    print('time.time() - start')
    print(time.time() - start)
    return trust_matrix, activity_matrix

def agree_trust_op(trainset, beta, epsilon,sim, ptype='user', istrainset=True, activity=False):
    print('======================== agree_trust |START|========================')
    start = time.time()
    cdef np.ndarray[np.double_t, ndim=2] trust_matrix, activity_matrix
    cdef int user_a, user_b, common_set_length, i, j, lenA, lenB, a_positive, b_positive, agreement, a_count, b_count, shorts_length
    cdef np.ndarray[np.double_t, ndim=1] a_ratings, b_ratings
    cdef np.ndarray[np.double_t, ndim=2] ratings
    cdef double a_val, b_val, trust, activity_val
    # cdef int[::1] intersect

    
    if istrainset == True:
        ratings = np.zeros((trainset.n_users, trainset.n_items))
        for u,i,r in trainset.all_ratings():
            ratings[u,i] =r
    else:
        ratings = trainset

    if ptype=='item':
        ratings = ratings.T

    n_x = ratings.shape[0]
    trust_matrix = np.zeros((n_x, n_x), np.double)
    activity_matrix = np.zeros((n_x, n_x), np.double)

    
    for user_a in range(n_x):
        trust_matrix[user_a, user_a] = 1
        for user_b in range(n_x):
        # for user_b in range(user_a + 1, n_x):
            a_ratings = ratings[user_a]
            b_ratings = ratings[user_b]
            # print('a_ratings')
            # print(a_ratings)
            # print('np.nonzero(a_ratings)')
            # print(np.nonzero(a_ratings))

            # a_ratings = a_ratings[np.nonzero(a_ratings)]
            # b_ratings = b_ratings[np.nonzero(b_ratings)]
            
            i = 0
            # j = 0
            common_set_length = 0
            # a_positive = 0
            # b_positive = 0
            lenA = a_ratings.shape[0]
            lenB = b_ratings.shape[0]
            a_count = 0
            b_count = 0

            agreement = 0

            # print('lenA')
            # print(lenA)
            # print('lenB')
            # print(lenB)
            # if lenA > lenB:
            #     shorts_length = lenB
            # else:
            #     shorts_length = lenA

            while (i < lenA):
                a_val = a_ratings[i]
                b_val = b_ratings[i]
                # print(a_val)
                # print(b_val)
                if a_val != 0 and b_val != 0:
                    common_set_length += 1
                    if a_val > beta and b_val > beta:
                        agreement += 1
                    elif a_val < beta and b_val < beta:
                        agreement += 1
                    elif a_val == beta and b_val == beta:#in ml-100k this will never be true for beta 2.5 since ratings are integers
                        agreement += 1
                elif a_val != 0:
                    a_count += 1
                elif b_val != 0:
                    b_count += 1
                i += 1

            # while (i < shorts_length):
            #     a_val = a_ratings[i]
            #     b_val = b_ratings[i]
            #     common_set_length += 1
            #     if a_val > beta and b_val > beta:
            #         agreement += 1
            #     elif a_val < beta and b_val < beta:
            #         agreement += 1
            #     elif a_val == beta and b_val == beta:#in ml-100k this will never be true for beta 2.5 since ratings are integers
            #         agreement += 1
            #     # else a_val != 0:
            #     #     a_count += 1
            #     # else b_val != 0:
            #     #     b_count += 1
            #     i += 1

            trust = 0
            activity_val = 0
            trust = agreement/(common_set_length+epsilon)
            # trust = np.sqrt(trust)
            # print(trust)
            # activity_val = a_count/(b_count+epsilon)
            activity_val = 1/(1+np.exp(-np.abs((a_count - b_count)/(a_count+b_count+epsilon))))
            # activity_val = 1/(1+np.exp(-(a_count - b_count)/(a_count+b_count+epsilon)))
            # activity_val = 1-1/(1+np.exp(-np.abs(a_count - b_count)))
            # activity_val = 1/(activity_val+0.9)
            # activity_matrix[user_a,user_b] = activity_val
            trust_matrix[user_a,user_b] = trust * sim[user_a,user_b] + activity_val
        activity_matrix[user_a, user_a] = 1
    # print('======================== agree_trust |END|========================')
    print('time.time() - start')
    print(time.time() - start)
    return trust_matrix

def agree_trust_old(trainset, beta, epsilon, ptype='user', istrainset=True, activity=False):
    print('======================== agree_trust_old |START|========================')
    if istrainset == True:
        ratings = np.zeros((trainset.n_users, trainset.n_items))
        for u,i,r in trainset.all_ratings():
            ratings[u,i] =r
    else:
        ratings = trainset

    if ptype=='item':
        ratings = ratings.T

    trust_matrix = np.zeros((ratings.shape[0], ratings.shape[0]))
    
    for user_a in range(ratings.shape[0]):
            for user_b in range(ratings.shape[0]):
                trust_matrix[user_b,user_b] = 1
                if user_a != user_b:
                    a_ratings = ratings[user_a]
                    b_ratings = ratings[user_b]

                    commonset = np.intersect1d(np.nonzero(a_ratings), np.nonzero(b_ratings))
                    
                    common_set_length = len(commonset)

                    trust = 0

                    if(common_set_length > 0):
                        a_positive = a_ratings[commonset] > beta
                        b_positive = b_ratings[commonset] > beta

                        agreement = np.sum(np.logical_not(np.logical_xor(a_positive, b_positive)))

                        # trust = agreement/common_set_length
                        trust = agreement/(common_set_length+epsilon)
                        # print('trust')
                        # print(trust)
                    # else:
                        # trust = (np.mean(ratings[user_a], dtype=np.float64) + np.mean(ratings[user_b], dtype=np.float64))/2
                        # print('mean')
                        # print(trust)
                    if activity == True:
                        # print("activity")
                        trust = trust*(len(np.nonzero(a_ratings))/(len(np.nonzero(a_ratings))+len(np.nonzero(b_ratings))))
                                               
                    trust_matrix[user_a,user_b] = trust
    # print('======================== agree_trust |END|========================')
    return trust_matrix

def odonovan_trust_old(trainset, algo, ptype='user', alpha=0.2):
    """Computes knn version of trust matrix proposed by J. O’Donovan and B. Smyth, in “Trust in recommender systems,” """
    print('======================== odonovan_trust |START|========================')
    
    cdef int rows = trainset.n_users
    cdef np.ndarray[np.double_t, ndim=2] trust_matrix,
    
    col_row_length = trainset.n_users

    if ptype == 'item':
        col_row_length = trainset.n_items

    trust_matrix = np.zeros((rows, rows))

    testset = trainset.build_testset()
    algo.fit(trainset)
    sim = algo.sim

    for x in range(rows):
        start = time.time()
        newset = cp.deepcopy(trainset)
        simc = cp.deepcopy(sim)
        simc[x] = 0
        if ptype == 'user':
            newset.ur[x] = []
        else:
            newset.ir[x] = []
    
        algo.fit(newset, simc)
        p = algo.test(testset)

        df = pd.DataFrame(p,columns=['uid', 'iid', 'rui', 'est', 'details'])

        df.sort_values(by=['uid'])
        df = df.loc[df['est'] != 0] #removes items predicted 0 
        df['err'] = abs(df.est - df.rui)

        filtered_df = df.loc[df['err'] < alpha] #alpha = 0.2

        uid1 = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts().keys().tolist()
        # new_list = [int(i)-1 for i in uid1]
        new_list = [newset.to_inner_uid(i)-1 for i in uid1]
       

        den = df.loc[df['uid'].isin(filtered_df.uid.unique())].uid.value_counts()
        

        uids = filtered_df.uid.value_counts().keys().tolist()
        
        nu = filtered_df.uid.value_counts()
        
        trust_matrix[x,new_list] = nu/den
        print('time.time() - start')
        print(time.time() - start)
        
    
    # print('======================== odonovan_trust |END|========================')
    return trust_matrix
