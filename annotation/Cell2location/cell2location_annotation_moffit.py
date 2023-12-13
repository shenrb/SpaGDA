#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.15               #
# ***********************************

import os
import scanpy as sc
import pandas as pd
import numpy as np
import copy
import time as tm
import json
import argparse
import anndata as ad

import cell2location

def fetch_ref(adata_ref):
    from cell2location.utils.filtering import filter_genes
    #selected = filter_genes(adata_ref, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12)

    # filter the object
    #adata_ref = adata_ref[:, selected].copy()
    
    # prepare anndata for the regression model
    cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
                            # 10X reaction / sample / batch
                            #batch_key='sequencing',
                            # cell type, covariate used for constructing signatures
                            labels_key='cell_types',
                            # multiplicative technical effects (platform, 3' vs 5', donor effect)
                            #categorical_covariate_keys=['Method']
                           )
    # create the regression model
    from cell2location.models import RegressionModel
    mod = RegressionModel(adata_ref)

    # view anndata_setup as a sanity check
    mod.view_anndata_setup()
    
    mod.train(max_epochs=100, use_gpu=True)
    
    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_ref = mod.export_posterior(
        adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
    )
    
    print("adata_ref shape:", adata_ref.shape)

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    return inf_aver


def run_cell2location(inf_aver, st_adata):

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=st_adata)

    # create and train the model
    mod = cell2location.models.Cell2location(
        st_adata, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=1,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=200
    )
    mod.view_anndata_setup()
    
    mod.train(max_epochs=4000,
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1,
          use_gpu=True)
    
    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    st_adata = mod.export_posterior(
        st_adata, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
    )
    
    # add 5% quantile, representing confident cell abundance, 'at least this amount is present',
    # to adata.obs with nice names for plotting
    st_adata.obs[st_adata.uns['mod']['factor_names']] = st_adata.obsm['q05_cell_abundance_w_sf']
    
    return st_adata



#palettes = dict(zip(st_adata.uns['mod']['factor_names'],['dodgerblue', 'lightgray','darkgreen','red', 'darkblue','chartreuse', 'orange']))
#st_adata.obsm["spatial"]=st_adata.obsm["spatial"][:,[1,0]]
#sc.pl.spatial(st_adata,color=["cell_types"],spot_size=60,palette=palettes)


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
    parser = argparse.ArgumentParser(description='run 5 folds cross validation for cell2location')
    parser.add_argument('--source_data', default='../Tangram/data/Moffit_RNA.h5ad', type=str, help='the reference scRNA dataset.')
    parser.add_argument('--target_dir', default='../Tangram/data', type=str, help='the dataset to be annotation.')
    parser.add_argument('--save_dir', default='Results', type=str, help='Save dir.')
    #parser.add_argument('--top_k', default=3000, type=int, help='Number of highly variable genes.')
    parser.add_argument('--sub_sections', default=4, type=int, help='Number of sub_sections.')
    args = parser.parse_args()

    sc.settings.figdir = args.save_dir

    # load the source dataset.
    RNA_adata = sc.read(args.source_data)
    RNA_adata.var_names_make_unique()

    #types_dict = dict(set(zip(RNA_adata.obs.cell_types.to_list(), RNA_adata.obs.labels.to_list())))
    
    annotation_list = list(pd.unique(RNA_adata.obs['cell_types']))
    annotation_list.sort()
    # load the target dataset.


    inf_aver = fetch_ref(RNA_adata)
    print('Finished reference scRNA dataset process...')

    samples = [1]
    for sid in samples:
        MERFISH_adata = sc.read(os.path.join(args.target_dir, 'MERFISH_%d.h5ad'%sid))
        MERFISH_adata.X = np.array(MERFISH_adata.X, dtype=np.int32)

        intersect = np.intersect1d(MERFISH_adata.var_names, inf_aver.index)
        MERFISH_adata = MERFISH_adata[:, intersect].copy()
        inf_aver2 = inf_aver.loc[intersect, :].copy()


        start = tm.time()
        spatial_cells = MERFISH_adata.shape[0] 
        tmp = []
        sections = [int(s) for s in np.linspace(0, spatial_cells, args.sub_sections+1)]
        for i in range(args.sub_sections):
            subsection_MERFISH_adata = MERFISH_adata[sections[i]:sections[i+1],:]

            #start = tm.time()
            st_adata = run_cell2location(inf_aver2, subsection_MERFISH_adata)
            results = st_adata.obsm['q05_cell_abundance_w_sf'].values
            type_list = st_adata.uns['mod']['factor_names']
            tmp.append(results)

        merged_results = np.vstack(tmp)

        predictions = np.array([int(i) for i in np.argmax(merged_results, axis=1)])
        gt = np.array([type_list.index(i) for i in MERFISH_adata.obs.cell_types])

        acc = sum(gt==predictions)*1. / MERFISH_adata.shape[0]
        print('ACC for sample %d: %.4f'%(sid, acc))
        print('Time cost: %d'%(tm.time()-start))


        with open('Results/acc.txt', 'a') as fs:
            fs.write('ACC for sample %d: %.4f\n'%(sid, acc))
            fs.write('Time cost: %d\n'%(tm.time()-start))

        result_df = pd.DataFrame(merged_results, columns=type_list, index=MERFISH_adata.obs_names.to_list())
        result_df.to_csv('Results/new_predictions_%d.csv'%sid)


