#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.08.10               #
# ***********************************

import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.io as io
import scanpy as sc
from anndata import AnnData
import json

# Prepare h5ad files for SC and ST datasets.

# SC dataset.
genes = pd.read_csv('/data/dataset/hypothalamic_merfish/GSE113576/genes.tsv',sep='\t',header=None)
barcodes = pd.read_csv('/data/dataset/hypothalamic_merfish/GSE113576/barcodes.tsv',sep='\t',header=None)
genes = np.array(genes.loc[:,1])
barcodes = np.array(barcodes.loc[:,0])
RNA_data = io.mmread('/data/dataset/hypothalamic_merfish/GSE113576/matrix.mtx')
RNA_data = RNA_data.todense()
RNA_data = pd.DataFrame(RNA_data,index=genes,columns=barcodes)

meta = pd.read_excel('/data/dataset/hypothalamic_merfish/sc/aau5324_moffitt_table-s1.xlsx',skiprows=range(0,1))

# filter lowely expressed genes
Genes_count = np.sum(RNA_data > 0, axis=1)
RNA_data = RNA_data.loc[Genes_count >=10,:]
del Genes_count

obs = pd.DataFrame(index=meta.iloc[:,0])
var = pd.DataFrame(index=RNA_data.index)

RNA_adata = AnnData(RNA_data.T.to_numpy(), obs=obs, var=var, dtype='float32')
RNA_adata.obs['cell_types'] = meta.iloc[:,3].to_list()

del_types = ['Ambiguous', 'Fibroblast', 'Unstable', 'Macrophage', 'Newly formed oligodendrocyte']
_RNA_adata = RNA_adata[~RNA_adata.obs['cell_types'].isin(del_types),:]

with open('/data/dataset/hypothalamic_merfish/selected_genes.json', 'r') as f:
    markers = json.load(f)

#_RNA_adata = _RNA_adata[:,_RNA_adata.var_names.isin(markers)]
_RNA_adata.obs['cell_types'] = _RNA_adata.obs['cell_types'].replace('Newly formed oligodendrocyte', 'Immature oligodendrocyte')

subclass_names = list(set(_RNA_adata.obs['cell_types']))
subclass_names.sort()
name2num_dict = dict(zip(subclass_names, range(len(subclass_names))))

_RNA_adata.obs['labels'] = [name2num_dict[i] for i in _RNA_adata.obs['cell_types']]
_RNA_adata.write('data/Moffit_RNA.h5ad')


# ST dataset.
MERFISH = pd.read_csv('/data/dataset/hypothalamic_merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv')
#Select  the 1st replicate, Naive state

samples = [1,2,7]

for sid in samples:
    MERFISH_sample = MERFISH.loc[MERFISH['Animal_ID']==sid,:]

    #remove Blank-1 to 5 and Fos --> 155 genes
    MERFISH_sample = MERFISH_sample.loc[MERFISH_sample['Cell_class']!='Ambiguous',:]
    MERFISH_meta = MERFISH_sample.iloc[:,0:9]
    MERFISH_data = MERFISH_sample.iloc[:,9:171]
    MERFISH_data = MERFISH_data.drop(columns = ['Blank_1','Blank_2','Blank_3','Blank_4','Blank_5','Fos'])

    obs = pd.DataFrame(index=MERFISH_data.index)
    var = pd.DataFrame(index=MERFISH_data.columns)

    MERFISH_adata = AnnData(MERFISH_data.to_numpy(), obs=obs, var=var, dtype='float32')

    ori = ['Endothelial 1', 'Endothelial 2', 'Endothelial 3', 'Astrocyte', 'OD Immature 1', 'OD Immature 2', 'OD Mature 1', 'OD Mature 2', 'OD Mature 3', 'OD Mature 4', 'Pericytes']
    tar = ['Endothelial', 'Endothelial', 'Endothelial', 'Astrocytes', 'Immature oligodendrocyte', 'Immature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mural']
    MERFISH_adata.obs['cell_types'] = MERFISH_meta['Cell_class'].replace(ori, tar).to_list()
    MERFISH_adata.obs['labels'] = [name2num_dict[i] for i in MERFISH_adata.obs['cell_types']]

    coord_3d = MERFISH_meta[['Bregma','Centroid_X','Centroid_Y']].to_numpy()
    coord_3d[:,0] = coord_3d[:,0]*(1000/0.22)
    MERFISH_adata.obsm['spatial'] = coord_3d # z,x,y

    MERFISH_adata.write('data/MERFISH_%d.h5ad'%sid)

