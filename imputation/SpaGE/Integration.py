#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.05.06               #
# ***********************************

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time as tm

with open ('SpaGE_pkl/MERFISH1.pkl', 'rb') as f:
    datadict = pickle.load(f)

MERFISH_data = datadict['MERFISH_data']
MERFISH_data_scaled = datadict['MERFISH_data_scaled']
MERFISH_meta = datadict['MERFISH_meta']
del datadict

with open ('SpaGE_pkl/Moffit_RNA.pkl', 'rb') as f:
    datadict = pickle.load(f)
    
RNA_data = datadict['RNA_data']
RNA_data_scaled = datadict['RNA_data_scaled']
del datadict

#### Leave One Out Validation ####
Common_data = RNA_data_scaled[np.intersect1d(MERFISH_data_scaled.columns,RNA_data_scaled.columns)]
print(Common_data.columns)
Imp_Genes = pd.DataFrame(columns=Common_data.columns)
precise_time = []
knn_time = []
for i in Common_data.columns:
    print(i)
    start = tm.time()
    from principal_vectors import PVComputation

    n_factors = 50
    n_pv = 50
    dim_reduction = 'pca'
    dim_reduction_target = 'pca'

    pv_FISH_RNA = PVComputation(n_factors = n_factors,n_pv = n_pv,dim_reduction = dim_reduction,dim_reduction_target = dim_reduction_target)

    pv_FISH_RNA.fit(Common_data.drop(i,axis=1),MERFISH_data_scaled[Common_data.columns].drop(i,axis=1))

    S = pv_FISH_RNA.source_components_.T
    
    Effective_n_pv = sum(np.diag(pv_FISH_RNA.cosine_similarity_matrix_) > 0.3)
    S = S[:,0:Effective_n_pv]

    Common_data_t = Common_data.drop(i,axis=1).dot(S)
    FISH_exp_t = MERFISH_data_scaled[Common_data.columns].drop(i,axis=1).dot(S)
    precise_time.append(tm.time()-start)
    
    start = tm.time()
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto',metric = 'cosine').fit(Common_data_t)
    distances, indices = nbrs.kneighbors(FISH_exp_t)

    Imp_Gene = np.zeros(MERFISH_data.shape[0])
    for j in range(0,MERFISH_data.shape[0]):
        weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]))
        weights = weights/(len(weights)-1)
        Imp_Gene[j] = np.sum(np.multiply(RNA_data[i][indices[j,:][distances[j,:] < 1]],weights))
    Imp_Gene[np.isnan(Imp_Gene)] = 0
    Imp_Genes[i] = Imp_Gene
    knn_time.append(tm.time()-start)

Imp_Genes.to_csv('Results/SpaGE_LeaveOneOut1.csv')
precise_time = pd.DataFrame(precise_time)
knn_time = pd.DataFrame(knn_time)
precise_time.to_csv('Results/SpaGE_PreciseTime1.csv', index = False)
knn_time.to_csv('Results/SpaGE_knnTime1.csv', index = False)