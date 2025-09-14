#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:05:11 2019

@author: ivan

Вычисление признаков относительно заданной траектории (в частности профиля).

"""

import numpy as np
from scipy import linalg
from scipy.spatial.distance import directed_hausdorff
import multiprocessing
import time


# Runners for multiprocessing
def runner(func, args):
    return func(*args)

def runfun(args):
    return runner(*args)


def genFeatures( begin, end,
        sim_in, force, ptime, pfl, af_inf, exceed_start, exceed,
        mu, ex_level, tf, dt,
        mA, mG, mCView, ind_m, n_jobs = multiprocessing.cpu_count()):

    sim = sim_in[(begin-ind_m):end,:]

    length = sim.shape[0]
    ind_m = int(ind_m)
    print('Creating pool with %d processes ...' % n_jobs, end='')
    pool = multiprocessing.Pool(n_jobs)
    print('ok')
    #  122 =~= (300000-30)/6/410 - count of tasks on core per second
    #   94 =~= (250000-30)/8/331
    print('Estimated time %.2f seconds'%(length/94/n_jobs))
    print('Start... ', time.strftime("%c"))
    start_time = time.time()

    tasks = \
      [(distPointToVec, (sim[i:i+1,:], pfl, ptime, mu)) for i in range(ind_m, length)] + \
      [(distVecToVec, (sim[i-ind_m:i,:], pfl)) for i in range(ind_m, length)] + \
      [(actFunc, (sim[i:i+1,:].T, ex_level, tf, mA, mG, mCView)) for i in range(ind_m, length)] + \
      [(hausdorf, (sim[i-ind_m:i,0:1], pfl[:,0:1])) for i in range(ind_m, length)] + \
      [(hausdorf, (sim[i-ind_m:i,0:2], pfl[:,0:2])) for i in range(ind_m, length)]

    feats_map = pool.map(runfun, tasks)
    print("--- %.2f seconds of parallel calc ---" % (time.time() - start_time))
    pool.close()
    pool.join()

    feats = np.hstack(( np.vstack(feats_map[:length-ind_m]),
                     np.vstack(feats_map[length-ind_m:2*(length-ind_m)]),
                     np.vstack(feats_map[2*(length-ind_m):3*(length-ind_m)]),
                     np.vstack(feats_map[3*(length-ind_m):4*(length-ind_m)]),
                     np.vstack(feats_map[4*(length-ind_m):5*(length-ind_m)])  ))

    # add Estimate of Time from window2Profile
    feats = np.hstack((feats,
            ptime[feats[:length-ind_m,8].astype(int),None]  ))

    # add action functionals
    feats = np.hstack(
        (feats,
         af_inf[feats[:,2].astype(int)], af_inf[feats[:,8].astype(int)] ))

    print('Finish ... ', time.strftime("%c"))
    return  feats



def loadSim(fname):
    sim = np.genfromtxt(fname=fname, delimiter=',', skip_header=1)
    return sim


def exampleGenerateFeatures():
    filename_sim = 'example_feat.csv'
#    filename_feat = 'out.txt'
    sim = np.genfromtxt(fname=filename_sim, delimiter=',', skip_header=1)


    feats = genFeatures(sim, pfl, ptime, mu, xf, tf, mA, mG, mCView,
                        ind_m, n_jobs = 8)

    return feats #, distVec2Vec, afs



if __name__ == '__main__':
    #distP2Vec, distVec2Vec, afs = exampleGenerateFeatures()
#    feats = exampleGenerateFeatures()
    pass
