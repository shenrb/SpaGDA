#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
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
import pickle
import scvi
from scvi.model import GIMVI
import scanpy as sc
import anndata as ad
import scipy.stats as st
from sklearn.metrics import mean_squared_error

def Normal_log1p(df):
    df = df.T
    cell_count = np.sum(df, axis=0)
    def Log_Norm(x):
        return np.log(((x/np.sum(x))*np.median(cell_count)).astype('float') + 1)

    df = df.apply(Log_Norm, axis=0)
    return df.T

if __name__ == '__main__':
    # Set arguments and hyper-parameters.
    parser = argparse.ArgumentParser(description='run 5 folds cross validation for gimVI')
    parser.add_argument('--source_data', default='data/Moffit_RNA.h5ad', type=str, help='the reference scRNA dataset.')
    parser.add_argument('--target_data', default='data/MERFISH1.h5ad', type=str, help='the dataset to be imputed.')
    parser.add_argument('--gene_groups', default='data/gene_groups.json', type=str, help='5 folds gene groups file.')
    parser.add_argument('--save_dir', default='Results', type=str, help='Save dir.')
    args = parser.parse_args()

    sc.settings.figdir = args.save_dir

    # load the target dataset.
    MERFISH_adata = sc.read(args.target_data)

    # load the source dataset.
    RNA_adata = sc.read(args.source_data)
    RNA_adata.var_names_make_unique()

    # load the splitted 5 folds gene groups for the shared genes.
    with open(args.gene_groups) as f:
        gene_groups = json.load(f)

    # Preparation
    raw_shared_gene = np.intersect1d(MERFISH_adata.var_names, RNA_adata.var_names)
    print(raw_shared_gene.shape, raw_shared_gene)
    Imputed_Genes = pd.DataFrame(columns=raw_shared_gene)

    #only use genes in both datasets
    common_MERFISH_adata = MERFISH_adata[:, raw_shared_gene].copy()
    common_RNA_adata = RNA_adata[:, raw_shared_gene].copy()
    sc.pp.filter_cells(common_RNA_adata, min_counts = 1)

    gimVI_time = []
    np.random.seed(6)

    # 5 folds cross validation.
    for k,v in gene_groups.items():
        print('==> Fold:', k)
        print('==> Imputed genes:', v)
        start = tm.time()

        #train_MERFISH_adata has a subset of the genes to train on
        train_MERFISH_adata = common_MERFISH_adata[:,~common_MERFISH_adata.var.index.isin(v)].copy()
        train_RNA_adata = common_RNA_adata.copy()

        scvi.data.setup_anndata(train_RNA_adata)
        scvi.data.setup_anndata(train_MERFISH_adata)

        model = GIMVI(train_RNA_adata, train_MERFISH_adata)
        model.train(n_epochs=200)

        _, fish_imputation = model.get_imputed_values(normalized=False)
        for gene in v:
            idx = list(raw_shared_gene).index(gene)
            Imputed_Genes[gene] = fish_imputation[:, idx]

        gimVI_time.append(tm.time()-start)

        #get the latent representations for the RNA and MERFISH data
        latent_RNA, latent_MERFISH = model.get_latent_representation()

        #concatenate to one latent representation
        latent_representation = np.concatenate([latent_RNA, latent_MERFISH])
        latent_adata = ad.AnnData(latent_representation)

        #labels which cells were from the RNA dataset and which were from the MERFISH dataset
        latent_labels = (['RNA'] * latent_RNA.shape[0]) + (['MERFISH'] * latent_MERFISH.shape[0])
        latent_adata.obs['labels'] = latent_labels

        #compute umap
        sc.pp.neighbors(latent_adata, use_rep = 'X')
        sc.tl.umap(latent_adata)

        #save umap representations to original RNA and MERFISH datasets
        train_RNA_adata.obsm['X_umap'] = latent_adata.obsm['X_umap'][:train_RNA_adata.shape[0]]
        train_MERFISH_adata.obsm['X_umap'] = latent_adata.obsm['X_umap'][train_RNA_adata.shape[0]:]

        #umap of the combined latent space
        sc.pl.umap(latent_adata, color = 'labels', save='_fold%s.pdf'%k)
        #umap of RNA dataset
        sc.pl.umap(train_RNA_adata, color = 'cell_types', save='_RNA_fold%s.pdf'%k)
        #umap of MERFISH dataset
        sc.pl.umap(train_MERFISH_adata, color = 'cell_types', save='_MERFISH_fold%s.pdf'%k)

    print("End of 5 fold cross validation:")
    print("==> Time of gimVI: ", gimVI_time)
    print("==> Mean time per fold: %.2f"%(sum(gimVI_time)/5))

    # metric
    gimVI_imputed = Imputed_Genes.loc[:,raw_shared_gene]
    MERFISH_data_shared = pd.DataFrame(common_MERFISH_adata.X, columns=raw_shared_gene)

    gimVI_SCC = pd.Series(index = raw_shared_gene)
    for i in raw_shared_gene:
        gimVI_SCC[i] = st.spearmanr(MERFISH_data_shared[i], gimVI_imputed[i])[0]

    gimVI_RMSE = mean_squared_error(MERFISH_data_shared.to_numpy(), gimVI_imputed.to_numpy())
    print("Imputation results of gimVI:")
    print("==> gimVI SCC median: %.4f"%gimVI_SCC.median())
    print("===> gimVI_RMSE: %.4f"%gimVI_RMSE)

    Imputed_Genes.to_csv('Results/gimVI_FiveFolds.csv')
    gimVI_time = pd.DataFrame(gimVI_time)
    gimVI_time.to_csv('Results/gimVI_Time.csv', index = False)
