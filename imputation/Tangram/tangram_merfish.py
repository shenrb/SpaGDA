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
import scanpy as sc
import anndata as ad
import tangram as tg
import scipy.stats as st
from sklearn.metrics import mean_squared_error

def Normal_log1p(df):
    df = df.T
    cell_count = np.sum(df, axis=0)
    def Log_Norm(x):
        return np.log(((x/np.sum(x))*np.median(cell_count)).astype('float') + 1)

    df = df.apply(Log_Norm, axis=0)
    return df.T

def select_top_variable_genes(data_mtx, top_k):
    """
    data_mtx: data matrix (cell by gene)
    top_k: number of highly variable genes to choose
    """
    var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind

if __name__ == '__main__':
    # Set arguments and hyper-parameters.
    parser = argparse.ArgumentParser(description='run 5 folds cross validation for Tangram')
    parser.add_argument('--source_data', default='../gimvi/data/Moffit_RNA.h5ad', type=str, help='the reference scRNA dataset.')
    parser.add_argument('--target_data', default='../gimvi/data/MERFISH1.h5ad', type=str, help='the dataset to be imputed.')
    parser.add_argument('--gene_groups', default='../gimvi/data/gene_groups.json', type=str, help='5 folds gene groups file.')
    parser.add_argument('--save_dir', default='Results', type=str, help='Save dir.')
    parser.add_argument('--sub_sections', default=4, type=int, help='Number of subsection of spatial data.')
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
    Imputed_Genes = pd.DataFrame(index=MERFISH_adata.obs_names.to_list(), columns=raw_shared_gene)

    tangram_time = []
    common_MERFISH_adata = MERFISH_adata[:, raw_shared_gene].copy()

    # 5 folds cross validation.
    for k,v in gene_groups.items():
        print('==> Fold:', k)
        print('==> Imputed genes:', v)
        start = tm.time()


        anchors = set(raw_shared_gene) - set(v)
        print('==> Anchors shape:', len(anchors))
        _RNA_adata = RNA_adata[:,RNA_adata.var_names.isin(list(raw_shared_gene))].copy()
        print('==> Reference RNA shape:', _RNA_adata.shape)

        # The spatial data has overmuch cells, thus split into 10 subsets.
        spatial_cells = common_MERFISH_adata.shape[0]   
        sections = [int(s) for s in np.linspace(0, spatial_cells, args.sub_sections+1)]
        tmp = []
        for i in range(args.sub_sections):
            subsection_common_MERFISH_adata = common_MERFISH_adata[sections[i]:sections[i+1],:]
            Normal_RNA_adata = _RNA_adata.copy()
            print('==> Subsection of Spatial dataset with shape:', subsection_common_MERFISH_adata.shape)
            tg.pp_adatas(Normal_RNA_adata, subsection_common_MERFISH_adata, genes=list(anchors))

            ad_map = tg.map_cells_to_space(Normal_RNA_adata, subsection_common_MERFISH_adata, mode="cells", density_prior='rna_count_based', num_epochs=400, device="cuda:0" )

            ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=Normal_RNA_adata)
            vlower = [g.lower() for g in v]
            print(len(set(ad_ge.var_names.to_list()) & set(vlower)), ad_ge.shape, )

            tangram_imputed = ad_ge[:, vlower]
            print('tangram_imputed shape:', tangram_imputed.shape, tangram_imputed.X.shape, Imputed_Genes.iloc[sections[i]:sections[i+1],:].shape, Imputed_Genes.iloc[sections[i]:sections[i+1],:][v].shape)
            tmp.append(tangram_imputed.X)

        Imputed_Genes[v] = np.vstack(tmp)
        tangram_time.append(tm.time()-start)

    print("End of 5 fold cross validation:")
    print("==> Time of tangram: ", tangram_time)
    print("==> Mean time per fold: %.2f"%(sum(tangram_time)/5))

    # metric
    tg_imputed = Imputed_Genes.loc[:,raw_shared_gene]
    MERFISH_data_shared = pd.DataFrame(common_MERFISH_adata.X, columns=raw_shared_gene)

    tangram_SCC = pd.Series(index = raw_shared_gene)
    for i in raw_shared_gene:
        tangram_SCC[i] = st.spearmanr(MERFISH_data_shared[i], tg_imputed[i])[0]

    print("Imputation results of tangram:")
    print("==> tangram SCC median: %.4f"%tangram_SCC.median())

    tangram_RMSE = mean_squared_error(MERFISH_data_shared.to_numpy(), tg_imputed.to_numpy())
    print("===> tangram_RMSE: %.4f"%tangram_RMSE)


    Imputed_Genes.to_csv('Results/tangram_FiveFolds.csv')
    tangram_time = pd.DataFrame(tangram_time)
    tangram_time.to_csv('Results/tangram_time.csv', index = False)
