#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2023.03.28               #
# ***********************************

import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.io as io
import scanpy as sc
from anndata import AnnData
import json


def select_top_variable_genes(data_mtx, top_k):
    """
    data_mtx: data matrix (cell by gene)
    top_k: number of highly variable genes to choose
    """
    var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind


# Prepare h5ad files for SC and ST datasets.
snRNA_file = '/data/bioinf/jyuan/rongboshen/datasets/snRNA/mop.h5ad'
merfish_mouse1_sample3 = '/data/bioinf/jyuan/rongboshen/datasets/mop/mouse1_sample3.h5ad'
sc_adata = sc.read(snRNA_file)
sc.pp.filter_genes(sc_adata, min_cells=1)

st_adata = sc.read(merfish_mouse1_sample3)
shared_genes = np.intersect1d(sc_adata.var_names, st_adata.var_names)
with open('data/shared_genes.json', 'w') as f:
    json.dump(list(shared_genes), f)

print(shared_genes.shape)

# SC dataset.
# filter lowely expressed genes
sc.pp.filter_cells(sc_adata, min_counts=10)

# high variable genes by Seurat
dif_sc_adata = sc_adata[:,~sc_adata.var_names.isin(shared_genes)]
sc.pp.highly_variable_genes(dif_sc_adata, n_top_genes=1000, flavor='seurat_v3')
sc_hvg_adata = dif_sc_adata[:, dif_sc_adata.var.highly_variable]

genes_df = sc_hvg_adata.var
genes_df = genes_df.sort_values('highly_variable_rank')

sc_hvgs = list(genes_df.index)
variance_norms = list(genes_df.variances_norm)
with open('data/sc_gene_variance_norm.json', 'w') as fs:
    json.dump(variance_norms, fs)

# Normal
sc.pp.normalize_total(sc_adata, target_sum=1e6)
sc.pp.log1p(sc_adata)

# select hvgs to imputation in this application.
downsample_df = sc_adata.obs.sample(frac=0.5, random_state=42)
downsamples = downsample_df.index.to_list()
sampled_sc_adata = sc_adata[sc_adata.obs.index.isin(downsamples),:]

Und_sc_adata = sampled_sc_adata[:,~sampled_sc_adata.var_names.isin(shared_genes)]
# high expr and high variable genes
hvg_idx = select_top_variable_genes(Und_sc_adata.X.toarray(), 1000) 
hvgs = Und_sc_adata.var_names[hvg_idx].to_list()

subclass_names = list(set(sc_adata.obs['Allen.subclass_label']))
subclass_names.sort()
name2num_dict = dict(zip(subclass_names, range(len(subclass_names))))


sc_adata.obs['cell_types'] = sc_adata.obs['Allen.subclass_label'].to_list()
sc_adata.obs['labels'] = [name2num_dict[i] for i in sc_adata.obs['cell_types']]

sc_adata.write('data/snRNA.h5ad')


# ST dataset.
st_adata = st_adata[:,st_adata.var_names.isin(shared_genes)]

sc.pp.normalize_total(st_adata)
sc.pp.log1p(st_adata)
st_adata.obs['subclass'] = st_adata.obs['subclass'].replace(['Micro'], ['Macrophage'])
st_adata.obs['cell_types'] = st_adata.obs['subclass'] 

st_adata.write('data/MERFISH.h5ad')

# hvgs to 10 FOLDs
folds = {}
for i in range(10):
    folds[i+1] = hvgs[100*i:100*(i+1)]
    print("fold %d: "%(i+1), hvgs[100*i:100*(i+1)])

with open('data/gene_folds.json', 'w') as fs:
    json.dump(folds, fs)


# sc_hvgs to 10 FOLDs
sc_folds = {}
for i in range(10):
    sc_folds[i+1] = sc_hvgs[100*i:100*(i+1)]
    print("fold %d: "%(i+1), sc_hvgs[100*i:100*(i+1)])

with open('data/sc_gene_folds.json', 'w') as fs:
    json.dump(sc_folds, fs)

