#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
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
import random

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



def anchor_genes_select(sc_hvgs, st_adata1, st_adata2, marker_genes, all_genes):

    sc.pp.normalize_total(st_adata1)
    sc.pp.log1p(st_adata1)

    sc.pp.normalize_total(st_adata2)
    sc.pp.log1p(st_adata2)


    hvg_idx = select_top_variable_genes(st_adata1.X, 1000)
    st_hvgs1 = st_adata1.var_names[hvg_idx].to_list()

    hvg_idx = select_top_variable_genes(st_adata2.X, 1000)
    st_hvgs2 = st_adata2.var_names[hvg_idx].to_list()

    hvg_anchors = list(set(sc_hvgs[:2000]) & set(st_hvgs1) & set(st_hvgs2))



    print('All marker genes %d:'%(len(marker_genes)),marker_genes)

    markers_in_anchor = list(set(hvg_anchors) & set(marker_genes))
    print('Marker genes in anchor gene %d:'%(len(markers_in_anchor)), markers_in_anchor)

    imputed_marker_genes = list(set(marker_genes) - set(hvg_anchors))
    print('Marker genes to impute %d:'%(len(imputed_marker_genes)), imputed_marker_genes)


    with open('data/imputed_marker_genes_v6.json', 'w') as fs:
        json.dump(imputed_marker_genes, fs)

    remain_sc_hvgs = []
    for g in sc_hvgs[:2000]:
        if g not in hvg_anchors:
            remain_sc_hvgs.append(g)

    target_genes = imputed_marker_genes
    for g in remain_sc_hvgs:
        if g not in target_genes:
            target_genes.append(g)
        if len(target_genes) == 1000:
            break

    return hvg_anchors, target_genes



# Prepare h5ad files for SC and ST datasets.
sc_file = '/data/bioinf/jyuan/rongboshen/datasets/testis/mouse_testes.h5ad'
slide_file1 = '/data/bioinf/jyuan/rongboshen/datasets/testis/WT3_seg.h5ad'
slide_file2 = '/data/bioinf/jyuan/rongboshen/datasets/testis/DM1_seg.h5ad'
sc_adata = sc.read(sc_file)
st_adata = sc.read(slide_file1)
st_adata2 = sc.read(slide_file2)

st_adata1 = processing(st_adata, min_genes=100, min_counts=200)
st_adata2 = processing(st_adata2, min_genes=100, min_counts=200)


sc_genes = sc_adata.var_names.tolist()
st_genes1 = st_adata1.var_names.tolist()
st_genes2 = st_adata2.var_names.tolist()

shared_genes = list(set(sc_genes) & set(st_genes1) & set(st_genes2))

with open('data/all_shared_genes2.json', 'w') as fs:
    json.dump(shared_genes, fs)

with open('data/sc_hvg_genes.json') as f:
    sc_hvgs = json.load(f)


# ST dataset.
shared_st_adata1 = st_adata1[:, shared_genes]
shared_st_adata2 = st_adata2[:, shared_genes]

marker_genes = {'ES':['Prm1', 'Prm2', 'Tnp1', 'Tnp2'], # all in shared genes
                'RS': ['Tssk1', 'Acrv1', 'Spaca1', 'Tsga8'], # all in imputed genes
                'SPC': ['Piwil1', 'Pttg1', 'Spag6', 'Tbpl1', 'Insl6'], #'Tbpl1' in shared genes, 'Piwil1', 'Spag6', 'Pttg1', 'Insl6' in imputed genes.
                'SPG': ['Uchl1', 'Sycp1', 'Stra8', 'Crabp1'], #'Sycp1' in shared genes, 'Uchl1' in imputed genes, 'Stra8', 'Crabp1' fail
                'Macrophage': ['Apoe', 'Dab2', 'Cd74', 'Adgre1'],
                'Endothelial': ['Vwf', 'Tie1', 'Tek', 'Ly6c1'],
                'Myoid': ['Acta2', 'Myh11', 'Myl6', 'Pdgfrb'],
                'Leydig': ['Cyp17a1', 'Cyp11a1', 'Star', 'Hsd3b1'],
                'Sertoli': ['Clu', 'Amhr2', 'Sox9', 'Ctsl', 'Rhox8']
                }

all_markers = []
for k,v in marker_genes.items():
    all_markers = all_markers + v

pseudotime_genes = ['mt-Nd1', 'Tuba3b', 'Stmn1', 'Cypt4', 'mt-Cytb', 'Hsp90aa1', 'Tnp2', 'Smcp', 'Gsg1', 'Oaz3', 'Hmgb4', 'Lyar', 'Prm1', 'Dbil5']

stage_genes = ['Tnp1', 'Prm1', 'Prm2', 'Serpina3a', 'Smcp', 'Ssxb1', 'Taf2', 'Pcaf', 'H2A', 'Ezh2', 'Brd8', 'Taf5', 'Trim24', 'Brd2']

all_markers = all_markers + pseudotime_genes + stage_genes

outside_genes = set(all_markers) - set(shared_genes)
print('Outside genes:', outside_genes)

add_genes = list(set(all_markers) - outside_genes)


anchor_genes, target_genes = anchor_genes_select(sc_hvgs, shared_st_adata1.copy(), shared_st_adata2.copy(), add_genes, shared_genes)
print('anchor genes:', len(anchor_genes))
with open('data/anchor_genes_v3.json', 'w') as fs:
    json.dump(anchor_genes, fs)


post_st_adata1 = shared_st_adata1[:, anchor_genes]
sc.pp.normalize_total(post_st_adata1)
sc.pp.log1p(post_st_adata1)
post_st_adata1.write('data/slide_WT3_v6.h5ad')

post_st_adata2 = shared_st_adata2[:, anchor_genes]
sc.pp.normalize_total(post_st_adata2)
sc.pp.log1p(post_st_adata2)
post_st_adata2.write('data/slide_DM1_v6.h5ad')

random.shuffle(target_genes)

folds = {}
for i in range(10):
    folds[i+1] = target_genes[100*i:100*(i+1)]
    print("fold %d: "%(i+1), target_genes[100*i:100*(i+1)])

with open('data/imputed_gene_folds_v6.json', 'w') as fs:
    json.dump(folds, fs)

