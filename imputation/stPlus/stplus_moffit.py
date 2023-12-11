#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.08.17               #
# ***********************************

import os
import numpy as np
import pandas as pd
import copy
import torch
import time as tm
import json
import argparse
import scanpy as sc
import anndata as ad
from stPlus import *  # pip3 install stPlus
import scipy.stats as st
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Set arguments and hyper-parameters.
    parser = argparse.ArgumentParser(description='run 5 folds cross validation for stPlus')
    parser.add_argument('--source_data', default='../gimvi/data/Moffit_RNA.h5ad', type=str, help='the reference scRNA dataset.')
    parser.add_argument('--target_data', default='../gimvi/data/MERFISH1.h5ad', type=str, help='the dataset to be imputed.')
    parser.add_argument('--gene_groups', default='data/gene_groups.json', type=str, help='5 folds gene groups file.')
    parser.add_argument('--save_dir', default='Results', type=str, help='Save dir.')
    args = parser.parse_args()

    sc.settings.figdir = args.save_dir

    # load the target dataset.
    MERFISH_adata = sc.read(args.target_data)

    # load the source dataset.
    RNA_adata = sc.read(args.source_data)
    RNA_adata.var_names_make_unique()


    types2label = {'Astrocytes':0, 'Endothelial':1, 'Ependymal':2, 'Excitatory':3, 'Immature oligodendrocyte':4, 'Inhibitory':5, 'Mature oligodendrocyte':6, 
                   'Microglia':7, 'Mural':8, 'Ambiguous':9, 'Fibroblast':10, 'Unstable':11, 'Macrophage':12,'Newly formed oligodendrocyte':13}

    MERFISH_adata.obs['labels'] = [types2label[x] for x in MERFISH_adata.obs['cell_types']]
    RNA_adata.obs['labels'] = [types2label[x] for x in RNA_adata.obs['cell_types']]

    # load the splitted 5 folds gene groups for the shared genes.
    with open(args.gene_groups) as f:
        gene_groups = json.load(f)

    # Preparation
    raw_shared_gene = np.intersect1d(MERFISH_adata.var_names, RNA_adata.var_names)
    print(raw_shared_gene.shape, raw_shared_gene)
    Imputed_Genes = pd.DataFrame(columns=raw_shared_gene)

    #only use genes in both datasets
    common_MERFISH_adata = MERFISH_adata[:, raw_shared_gene].copy()
    Normal_RNA_adata = RNA_adata.copy()

    RNA_df = pd.DataFrame(Normal_RNA_adata.X, index=Normal_RNA_adata.obs_names.to_list(), columns=Normal_RNA_adata.var_names.to_list())

    stPlus_time = []

    # 5 folds cross validation.
    for k,v in gene_groups.items():
        print('==> Fold:', k)
        print('==> Imputed genes:', v)
        start = tm.time()

        #train_MERFISH_adata has a subset of the genes to train on
        train_MERFISH_adata = common_MERFISH_adata[:,~common_MERFISH_adata.var.index.isin(v)].copy()

        MERFISH_df = pd.DataFrame(train_MERFISH_adata.X, index=train_MERFISH_adata.obs_names.to_list(), columns=train_MERFISH_adata.var_names.to_list())
 
        save_path_prefix = 'model/stPlus-fold%s'%k
        stPlus_res = stPlus(MERFISH_df, RNA_df, v, save_path_prefix, converge_ratio=0.001)
        Imputed_Genes[stPlus_res.columns.values] = stPlus_res

        stPlus_time.append(tm.time()-start)

    print("End of 5 fold cross validation:")
    print("==> Time of stPlus: ", stPlus_time)
    print("==> Mean time per fold: %.2f"%(sum(stPlus_time)/5))

    # metric
    stPlus_imputed = Imputed_Genes.loc[:,raw_shared_gene]
    MERFISH_data_shared = pd.DataFrame(common_MERFISH_adata.X, columns=raw_shared_gene)

    stPlus_SCC = pd.Series(index = raw_shared_gene)
    for i in raw_shared_gene:
        stPlus_SCC[i] = st.spearmanr(MERFISH_data_shared[i], stPlus_imputed[i])[0]

    stPlus_RMSE = mean_squared_error(MERFISH_data_shared.to_numpy(), stPlus_imputed.to_numpy())
    print("Imputation results of stPlus:")
    print("==> stPlus SCC median: %.4f"%stPlus_SCC.median())
    print("===> stPlus_RMSE: %.4f"%stPlus_RMSE)

    Imputed_Genes.to_csv('Results/stPlus_FiveFolds.csv')
    stPlus_time = pd.DataFrame(stPlus_time)
    stPlus_time.to_csv('Results/stPlus_Time.csv', index = False)
