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
    #parser.add_argument('--top_k', default=3000, type=int, help='Number of highly variable genes.')
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

    #Undetected_RNA_adata = RNA_adata[:,~RNA_adata.var_names.isin(raw_shared_gene)]
    #hvg_idx = select_top_variable_genes(Undetected_RNA_adata.X, args.top_k)
    #hvgs = Undetected_RNA_adata.var_names[hvg_idx].to_list()

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

            # training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.
            # overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.
            # uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata.
            # rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata.
            tg.pp_adatas(Normal_RNA_adata, subsection_common_MERFISH_adata, genes=list(anchors))

            ad_map = tg.map_cells_to_space(Normal_RNA_adata, subsection_common_MERFISH_adata, mode="cells", density_prior='rna_count_based', num_epochs=400, device="cuda:0" )
            # mode="clusters", density_prior='uniform'
            # cluster_label='cell_subclass',  # .obs field w cell types
            # device='cpu',device="cuda:0"
            # The mapping results are stored in the returned AnnData structure, saved as ad_map, structured as following: 
            # -> The cell-by-spot matrix X contains the probability of cell i to be in spot j. 
            # -> The obs dataframe contains the metadata of the single cells. 
            # -> The var dataframe contains the metadata of the spatial data. 
            # -> The uns dictionary contains a dataframe with various information about the training genes (saved as train_genes_df).

            # If the mapping mode is 'cells', we can now generate the “new spatial data” using the mapped single cell: this is done via project_genes. 
            # -> The function accepts as input a mapping (adata_map) and corresponding single cell data (adata_sc). 
            # -> The result is a voxel-by-gene AnnData, formally similar to adata_st, but containing gene expression from the mapped single cell data rather than Visium. 
            # -> For downstream analysis, we always replace adata_st with the corresponding ad_ge.

            ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=Normal_RNA_adata)
            vlower = [g.lower() for g in v]
            print(len(set(ad_ge.var_names.to_list()) & set(vlower)), ad_ge.shape, )
            #print(ad_ge.var_names.to_list())

            #df_all_genes = tg.compare_spatial_geneexp(ad_ge, subsection_common_MERFISH_adata, Normal_RNA_adata)

            # It is convenient to compute the similarity scores of all genes, which can be done by compare_spatial_geneexp. 
            # This function accepts two spatial AnnDatas (ie voxel-by-gene), and returns a dataframe with simlarity scores for all genes. Training genes are flagged by the boolean field is_training. 
            # If we also pass single cell AnnData to compare_spatial_geneexp function like below, 
            #    a dataframe with additional sparsity columns - sparsity_sc (single cell data sparsity) and sparsity_diff (spatial data sparsity - single cell data sparsity) will return. 
            #    This is required if we want to call plot_test_scores function later with the returned datafrme from compare_spatial_geneexp function.
            #ad_ge.write('tmp.h5ad')
            tangram_imputed = ad_ge[:, vlower]
            print('tangram_imputed shape:', tangram_imputed.shape, tangram_imputed.X.shape, Imputed_Genes.iloc[sections[i]:sections[i+1],:].shape, Imputed_Genes.iloc[sections[i]:sections[i+1],:][v].shape)
            tmp.append(tangram_imputed.X)

        #Imputed_Genes.iloc[sections[i]:sections[i+1],:][v] = tangram_imputed.X
        Imputed_Genes[v] = np.vstack(tmp)
        tangram_time.append(tm.time()-start)

    print("End of 5 fold cross validation:")
    print("==> Time of tangram: ", tangram_time)
    print("==> Mean time per fold: %.2f"%(sum(tangram_time)/5))

    #Imputed_Genes.to_csv('Results/tangram_FiveFolds.csv')
    # metric
    tg_imputed = Imputed_Genes.loc[:,raw_shared_gene]
    MERFISH_data_shared = pd.DataFrame(common_MERFISH_adata.X, columns=raw_shared_gene)

    #MERFISH_data_shared.to_csv('Results/MERFISH_ori.csv')

    tangram_SCC = pd.Series(index = raw_shared_gene)
    for i in raw_shared_gene:
        tangram_SCC[i] = st.spearmanr(MERFISH_data_shared[i], tg_imputed[i])[0]

    # Normalization for fair comparsion of RMSE with other methods.
    #normal_tg_imputed = Normal_log1p(tg_imputed)
    #normal_MERFISH_data_shared = Normal_log1p(MERFISH_data_shared)

    print("Imputation results of tangram:")
    print("==> tangram SCC median: %.4f"%tangram_SCC.median())

    tangram_RMSE = mean_squared_error(MERFISH_data_shared.to_numpy(), tg_imputed.to_numpy())
    #tangram_RMSE = mean_squared_error(normal_MERFISH_data_shared.to_numpy(), normal_tg_imputed.to_numpy())
    #print("Imputation results of tangram:")
    #print("==> tangram SCC median: %.4f"%tangram_SCC.median())
    print("===> tangram_RMSE: %.4f"%tangram_RMSE)


    Imputed_Genes.to_csv('Results/tangram_FiveFolds.csv')
    tangram_time = pd.DataFrame(tangram_time)
    tangram_time.to_csv('Results/tangram_time.csv', index = False)
