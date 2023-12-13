#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.10               #
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
    parser = argparse.ArgumentParser(description='run 5 folds cross validation for tangram')
    parser.add_argument('--source_data', default='data/Moffit_RNA.h5ad', type=str, help='the reference scRNA dataset.')
    parser.add_argument('--target_dir', default='data', type=str, help='the dataset to be annotation.')
    parser.add_argument('--save_dir', default='Results', type=str, help='Save dir.')
    #parser.add_argument('--top_k', default=3000, type=int, help='Number of highly variable genes.')
    parser.add_argument('--sub_sections', default=4, type=int, help='Number of sub_sections.')
    args = parser.parse_args()

    sc.settings.figdir = args.save_dir

    # load the source dataset.
    RNA_adata = sc.read(args.source_data)
    RNA_adata.var_names_make_unique()

    #sc.pp.normalize_total(RNA_adata)
    #sc.pp.log1p(RNA_adata)

    #types_dict = dict(set(zip(RNA_adata.obs.cell_types.to_list(), RNA_adata.obs.labels.to_list())))
    
    annotation_list = list(pd.unique(RNA_adata.obs['cell_types']))
    annotation_list.sort()
    # load the target dataset.

    samples = [1]
    tangram_time = []
    for sid in samples:
        MERFISH_adata = sc.read(os.path.join(args.target_dir, 'MERFISH_%d.h5ad'%sid))
        #sc.pp.normalize_total(MERFISH_adata)
        #sc.pp.log1p(MERFISH_adata)

        # Preparation
        raw_shared_gene = np.intersect1d(MERFISH_adata.var_names, RNA_adata.var_names)
        print(raw_shared_gene.shape, raw_shared_gene)


        #Undetected_RNA_adata = RNA_adata[:,~RNA_adata.var_names.isin(raw_shared_gene)]
        #hvg_idx = select_top_variable_genes(Undetected_RNA_adata.X, args.top_k)
        #hvgs = Undetected_RNA_adata.var_names[hvg_idx].to_list()

        print('==> sample:', sid)
        start = tm.time()

        common_MERFISH_adata = MERFISH_adata[:, raw_shared_gene].copy()
        anchors = set(raw_shared_gene)
        print('==> Anchors shape:', len(anchors))
        _RNA_adata = RNA_adata[:,RNA_adata.var_names.isin(list(raw_shared_gene))].copy()
        #_RNA_adata = RNA_adata.copy()
        print('==> Reference RNA shape:', _RNA_adata.shape)

        spatial_cells = common_MERFISH_adata.shape[0] 

        tmp = []
        sections = [int(s) for s in np.linspace(0, spatial_cells, args.sub_sections+1)]
        for i in range(args.sub_sections):
            subsection_common_MERFISH_adata = common_MERFISH_adata[sections[i]:sections[i+1],:]
            Normal_RNA_adata = _RNA_adata.copy()
            print('==> Subsection of Spatial dataset with shape:', subsection_common_MERFISH_adata.shape)

            tg.pp_adatas(Normal_RNA_adata, subsection_common_MERFISH_adata, genes=list(anchors))
            ad_map = tg.map_cells_to_space(Normal_RNA_adata, subsection_common_MERFISH_adata, mode="cells", 
                                           density_prior='rna_count_based', num_epochs=400, device="cuda:0" )
            tg.project_cell_annotations(ad_map, subsection_common_MERFISH_adata, annotation="cell_types")

            preds = subsection_common_MERFISH_adata.obsm["tangram_ct_pred"][annotation_list].to_numpy()
            tmp.append(preds)
    

        #tg.pp_adatas(_RNA_adata, common_MERFISH_adata, genes=list(anchors))
        #ad_map = tg.map_cells_to_space(_RNA_adata, common_MERFISH_adata, mode="cells", density_prior='rna_count_based', num_epochs=400, device="cuda:0" )

        #tg.project_cell_annotations(ad_map, common_MERFISH_adata, annotation="cell_types")
        #results = common_MERFISH_adata.obsm["tangram_ct_pred"][annotation_list].to_numpy()

        results = np.vstack(tmp)
        predictions = np.array([int(i) for i in np.argmax(results, axis=1)])
        gt = np.array(common_MERFISH_adata.obs.labels)

        acc = sum(gt==predictions)*1. / spatial_cells
        print('ACC for sample %d: %.4f'%(sid, acc))
        print('Time cost: %d'%(tm.time()-start))
        tangram_time.append(tm.time()-start)

        result_df = pd.DataFrame(results, columns=annotation_list, index=common_MERFISH_adata.obs_names.to_list())
        result_df.to_csv('Results/predictions_%d.csv'%sid)

