#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.23               #
# ***********************************

import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.io as io
import scanpy as sc
from anndata import AnnData
import json

def processing(adata, min_genes=200, min_counts=300, up_filter=False):
    # basic filter.
    print(f"#cells before filter: {adata.n_obs}")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    if up_filter:
        max_counts=np.percentile(adata.X.sum(1),99.0)
        sc.pp.filter_cells(adata, max_counts=max_counts)
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"#cells after basic filter: {adata.n_obs}")

    # QC
    adata.var['mt'] = adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    print(f"#cells after MT filter: {adata.n_obs}")

    return adata

def select_top_variable_genes(data_mtx, top_k):
    """
    data_mtx: data matrix (cell by gene)
    top_k: number of highly variable genes to choose
    """
    var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind


def select_high_variable_genes(adata, top_k, sorted=True):

    # normalize 
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # hvgs
    #sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=top_k)
    #hvgs = adata.var[adata.var.highly_variable].index.to_list()

    hvg_idx = select_top_variable_genes(adata.X, top_k)
    hvgs = adata.var_names[hvg_idx].to_list()

    if sorted:
        hvgs.sort()

    return hvgs


def anchor_genes_select(sc_adata, st_adata1, st_adata2, marker_genes, all_genes):

    sc.pp.normalize_total(sc_adata, target_sum=1e6)
    sc.pp.log1p(sc_adata)

    sc.pp.normalize_total(st_adata1)
    sc.pp.log1p(st_adata1)

    sc.pp.normalize_total(st_adata2)
    sc.pp.log1p(st_adata2)

    hvg_idx = select_top_variable_genes(sc_adata.X, 2000)
    sc_hvgs = sc_adata.var_names[hvg_idx].to_list()

    hvg_idx = select_top_variable_genes(st_adata1.X, 1000)
    st_hvgs1 = st_adata1.var_names[hvg_idx].to_list()

    hvg_idx = select_top_variable_genes(st_adata2.X, 1000)
    st_hvgs2 = st_adata2.var_names[hvg_idx].to_list()

    hvg_anchors = list(set(sc_hvgs) & set(st_hvgs1) & set(st_hvgs2))

    tmp = []
    for k,v in marker_genes.items():
        tmp = tmp + v
    tmp = set(tmp) & set(all_genes)

    hvg_anchors = hvg_anchors + list(tmp)

    return list(set(hvg_anchors)), sc_adata, st_adata1, st_adata2



# Prepare h5ad files for SC datasets.
scRNA_file = '/data/bioinf/jyuan/rongboshen/datasets/testis/mouse_testes.h5ad'
sc_adata = sc.read(scRNA_file)

sc_adata = processing(sc_adata, min_genes=200, min_counts=300)

with open('data/all_shared_genes.json') as f:
    shared_genes = json.load(f)

# SC dataset.
shared_sc_adata = sc_adata[:, shared_genes]
shared_sc_adata2 = shared_sc_adata.copy()
sc.pp.normalize_total(shared_sc_adata)
sc.pp.log1p(shared_sc_adata)
shared_sc_adata.write('data/testes_benchmark.h5ad')

hvg_idx = select_top_variable_genes(shared_sc_adata.X, 5000)
sc_hvgs = shared_sc_adata.var_names[hvg_idx].to_list()

with open('data/sc_hvg_genes_benchmark.json', 'w') as f:
    json.dump(sc_hvgs, f)

sc.pp.normalize_total(shared_sc_adata2, target_sum=1e6)
sc.pp.log1p(shared_sc_adata2)
shared_sc_adata2.write('data/testes_benchmark_1e6.h5ad')

