#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.07.06               #
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

def Log_Norm(x):
    return np.log(((x/np.sum(x))*1000000) + 1)

#RNA_data = RNA_data.apply(Log_Norm,axis=0)
#RNA_data_scaled = pd.DataFrame(data=st.zscore(RNA_data.T),index = RNA_data.columns,columns=RNA_data.index)

obs = pd.DataFrame(index=meta.iloc[:,0])
var = pd.DataFrame(index=RNA_data.index)

RNA_adata = AnnData(RNA_data.T.to_numpy(), obs=obs, var=var, dtype='float32')
#RNA_adata_scaled = AnnData(RNA_data_scaled.to_numpy(), obs=obs, var=var, dtype='float32')

RNA_adata.obs['cell_types'] = meta.iloc[:,3].to_list()

del_types = ['Ambiguous', 'Fibroblast', 'Unstable', 'Macrophage', 'Newly formed oligodendrocyte']
upsample_types = ['Ependymal']

_RNA_adata = RNA_adata[~RNA_adata.obs['cell_types'].isin(del_types),:]

with open('/data/dataset/hypothalamic_merfish/selected_genes.json', 'r') as f:
    markers = json.load(f)

_RNA_adata = _RNA_adata[:,_RNA_adata.var_names.isin(markers)]
_RNA_adata.obs['cell_types'] = _RNA_adata.obs['cell_types'].replace('Newly formed oligodendrocyte', 'Immature oligodendrocyte')

subclass_names = list(set(_RNA_adata.obs['cell_types']))
subclass_names.sort()
name2num_dict = dict(zip(subclass_names, range(len(subclass_names))))


adata_upsample = _RNA_adata[_RNA_adata.obs.cell_types.isin(upsample_types),:]
adata_normal = _RNA_adata[~_RNA_adata.obs.cell_types.isin(upsample_types),:]

features_upsample = np.array(adata_upsample.X, dtype=np.float32)
labels_upsample = adata_upsample.obs.cell_types.to_list()

features_normal = np.array(adata_normal.X, dtype=np.float32)
labels_normal = adata_normal.obs.cell_types.to_list()

features_up = np.tile(features_upsample, (20,1))
shape = features_up.shape
noise = np.random.rand(shape[0],shape[1])
features_up = features_up + 0.01*noise.astype(np.float32)*np.mean(features_up)
labels_up = labels_upsample*20

features = np.concatenate((features_normal,features_up), axis=0)
for i in range(features.shape[0]):
    norm = np.linalg.norm(features[i])
    if norm != 0:
        features[i] = features[i]/norm
subclass_labels = labels_normal + labels_up

obs = pd.DataFrame(index=range(features.shape[0]))
var = pd.DataFrame(index=_RNA_adata.var_names.to_list())
Post_RNA_adata = AnnData(features, obs=obs, var=var, dtype='float32')
Post_RNA_adata.obs['cell_types'] = subclass_labels
Post_RNA_adata.obs['labels'] = [name2num_dict[i] for i in subclass_labels]

share_genes = Post_RNA_adata.var_names.to_list()
_Post_RNA_adata = Post_RNA_adata[:,share_genes].copy()

_Post_RNA_adata.write('data/Moffit_RNA.h5ad')
#RNA_adata_scaled.write('data/Moffit_RNA_scaled.h5ad')


# ST dataset.

MERFISH = pd.read_csv('/data/dataset/hypothalamic_merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv')
#Select  the 1st replicate, Naive state

MERFISH_1 = MERFISH.loc[MERFISH['Animal_ID']==1,:]

#remove Blank-1 to 5 and Fos --> 155 genes
MERFISH_1 = MERFISH_1.loc[MERFISH_1['Cell_class']!='Ambiguous',:]
MERFISH_meta = MERFISH_1.iloc[:,0:9]
MERFISH_data = MERFISH_1.iloc[:,9:171]
MERFISH_data = MERFISH_data.drop(columns = ['Blank_1','Blank_2','Blank_3','Blank_4','Blank_5','Fos'])

MERFISH_data = MERFISH_data.T
cell_count = np.sum(MERFISH_data,axis=0)
def Log_Norm(x):
    return np.log(((x/np.sum(x))*np.median(cell_count)) + 1)

#MERFISH_data = MERFISH_data.apply(Log_Norm,axis=0)
#MERFISH_data_scaled = pd.DataFrame(data=st.zscore(MERFISH_data.T),index = MERFISH_data.columns,columns=MERFISH_data.index)

obs = pd.DataFrame(index=MERFISH_data.columns)
var = pd.DataFrame(index=MERFISH_data.index)

features = MERFISH_data.T.to_numpy()
for i in range(features.shape[0]):
    norm = np.linalg.norm(features[i])
    if norm != 0:
        features[i] = features[i]/norm

MERFISH_adata = AnnData(features, obs=obs, var=var, dtype='float32')
MERFISH_adata = MERFISH_adata[:,share_genes]
#MERFISH_adata_scaled = AnnData(MERFISH_data_scaled.to_numpy(), obs=obs, var=var, dtype='float32')

ori = ['Endothelial 1', 'Endothelial 2', 'Endothelial 3', 'Astrocyte', 'OD Immature 1', 'OD Immature 2', 'OD Mature 1', 'OD Mature 2', 'OD Mature 3', 'OD Mature 4', 'Pericytes']
tar = ['Endothelial', 'Endothelial', 'Endothelial', 'Astrocytes', 'Immature oligodendrocyte', 'Immature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mural']
MERFISH_adata.obs['cell_types'] = MERFISH_meta['Cell_class'].replace(ori, tar).to_list()
MERFISH_adata.obs['labels'] = [name2num_dict[i] for i in MERFISH_adata.obs['cell_types']]

coord_3d = MERFISH_meta[['Bregma','Centroid_X','Centroid_Y']].to_numpy()
coord_3d[:,0] = coord_3d[:,0]*(1000/0.22)
MERFISH_adata.obsm['spatial'] = coord_3d # z,x,y

MERFISH_adata.write('data/MERFISH.h5ad')
#MERFISH_adata_scaled.write('data/MERFISH_scaled.h5ad')
