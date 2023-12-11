#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.17               #
# ***********************************

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time as tm
import json
import argparse
import scipy.stats as st
from sklearn.metrics import mean_squared_error

from principal_vectors import PVComputation

if __name__ == '__main__':
    # Set arguments and hyper-parameters.
    parser = argparse.ArgumentParser(description='run 5 folds cross validation for SpaGE')
    parser.add_argument('--source_data', default='data/Moffit_RNA.pkl', type=str, help='the reference scRNA dataset.')
    parser.add_argument('--target_data', default='data/MERFISH1.pkl', type=str, help='the dataset to be imputed.')
    parser.add_argument('--gene_groups', default='data/gene_groups.json', type=str, help='5 folds gene groups file.')
    args = parser.parse_args()

    # load the target dataset.
    with open (args.target_data, 'rb') as f:
        datadict = pickle.load(f)
    MERFISH_data = datadict['MERFISH_data']
    MERFISH_data_scaled = datadict['MERFISH_data_scaled']
    MERFISH_meta = datadict['MERFISH_meta']
    del datadict

    # load the source dataset.
    with open (args.source_data, 'rb') as f:
        datadict = pickle.load(f)
    RNA_data = datadict['RNA_data']
    RNA_data_scaled = datadict['RNA_data_scaled']
    del datadict

    # load the splitted 5 folds gene groups for the shared genes.
    with open(args.gene_groups) as f:
        gene_groups = json.load(f)

    # Preparation
    raw_shared_gene = np.intersect1d(MERFISH_data_scaled.columns, RNA_data_scaled.columns)
    print(raw_shared_gene.shape, raw_shared_gene)
    Imputed_Genes = pd.DataFrame(columns=raw_shared_gene)
    precise_time = []
    knn_time = []

    np.random.seed(6)

    # 5 folds cross validation.
    for k,v in gene_groups.items():
        print('==> Fold:', k)
        print('==> Imputed genes:', v)
        start = tm.time()

        pv_FISH_RNA = PVComputation(n_factors = 50, n_pv = 50, dim_reduction = 'pca', dim_reduction_target = 'pca')
        pv_FISH_RNA.fit(RNA_data_scaled[raw_shared_gene].drop(v,axis=1), MERFISH_data_scaled[raw_shared_gene].drop(v,axis=1))

        S = pv_FISH_RNA.source_components_.T
        Effective_n_pv = sum(np.diag(pv_FISH_RNA.cosine_similarity_matrix_) > 0.3)
        S = S[:,0:Effective_n_pv]

        Common_data_t = RNA_data_scaled[raw_shared_gene].drop(v,axis=1).dot(S)
        FISH_exp_t = MERFISH_data_scaled[raw_shared_gene].drop(v,axis=1).dot(S)
        precise_time.append(tm.time()-start)

        start = tm.time()
        nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto',metric = 'cosine').fit(Common_data_t)
        distances, indices = nbrs.kneighbors(FISH_exp_t)

        Imp_Gene = np.zeros((MERFISH_data.shape[0], len(v)))
        source_RNA_data = RNA_data[v]
        for j in range(0, MERFISH_data.shape[0]):
            weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]))
            weights = weights/(len(weights)-1)
            Imp_Gene[j] = np.array(np.sum(np.multiply(source_RNA_data.iloc[indices[j,:][distances[j,:] < 1],:].T,weights),axis=1))
            print('[Num %6d/ %6d] \r'% (j + 1, MERFISH_data.shape[0]), end='')
        Imp_Gene[np.isnan(Imp_Gene)] = 0

        for i, gene in enumerate(v):
            Imputed_Genes[gene] = Imp_Gene[:,i]

        knn_time.append(tm.time()-start)

    print("End of 5 fold cross validation:")
    print("==> Time of PRECISE: ", precise_time)
    print("==> Time of KNN: ", knn_time)
    print("==> Mean time per fold: %.2f"%((sum(precise_time)+sum(knn_time))/5))

    # metric
    SpaGE_imputed = Imputed_Genes.loc[:,raw_shared_gene]
    MERFISH_data_shared = MERFISH_data.loc[:,raw_shared_gene]

    SpaGE_SCC = pd.Series(index = raw_shared_gene)
    for i in raw_shared_gene:
        SpaGE_SCC[i] = st.spearmanr(MERFISH_data_shared[i], SpaGE_imputed[i])[0]

    SpaGE_RMSE = mean_squared_error(MERFISH_data_shared.to_numpy(), SpaGE_imputed.to_numpy())
    print("Imputation results of SpaGE:")
    print("==> SpaGE SCC median: %.4f"%SpaGE_SCC.median())
    print("===> SpaGE_RMSE: %.4f"%SpaGE_RMSE)

    Imputed_Genes.to_csv('Results/SpaGE_FiveFolds.csv')
    precise_time = pd.DataFrame(precise_time)
    knn_time = pd.DataFrame(knn_time)
    precise_time.to_csv('Results/SpaGE_PreciseTime.csv', index = False)
    knn_time.to_csv('Results/SpaGE_knnTime.csv', index = False)
