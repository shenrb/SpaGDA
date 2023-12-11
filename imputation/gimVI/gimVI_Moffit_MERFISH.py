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
import scipy.stats as st
import pickle
import scipy.io as io
import scanpy as sc
from anndata import AnnData

# scRNA dataset: Moffit
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

RNA_data = RNA_data.apply(Log_Norm,axis=0)
obs = pd.DataFrame(index=meta.iloc[:,0])
var = pd.DataFrame(index=RNA_data.index)

RNA_adata = AnnData(RNA_data.T.to_numpy(), obs=obs, var=var, dtype='float32')
RNA_adata.obs['cell_types'] = meta.iloc[:,3].to_list()
RNA_adata.write('data/Moffit_RNA.h5ad')

# Spatial dataset: MERFISH
MERFISH = pd.read_csv('/data/dataset/hypothalamic_merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv')
#Select the 1st replicate, Naive state
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

MERFISH_data = MERFISH_data.apply(Log_Norm,axis=0)
obs = pd.DataFrame(index=MERFISH_data.columns)
var = pd.DataFrame(index=MERFISH_data.index)

MERFISH_adata = AnnData(MERFISH_data.T.to_numpy(), obs=obs, var=var, dtype='float32')

ori = ['Endothelial 1', 'Endothelial 2', 'Endothelial 3', 'Astrocyte', 'OD Immature 1', 'OD Immature 2', 'OD Mature 1', 'OD Mature 2', 'OD Mature 3', 'OD Mature 4', 'Pericytes']
tar = ['Endothelial', 'Endothelial', 'Endothelial', 'Astrocytes', 'Immature oligodendrocyte', 'Immature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mural']

MERFISH_adata.obs['cell_types'] = MERFISH_meta['Cell_class'].replace(ori, tar).to_list()
MERFISH_adata.write('data/MERFISH1.h5ad')

