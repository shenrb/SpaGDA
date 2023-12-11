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

# filter lowely expressed genes
Genes_count = np.sum(RNA_data > 0, axis=1)
RNA_data = RNA_data.loc[Genes_count >=10,:]
del Genes_count

def Log_Norm(x):
    return np.log(((x/np.sum(x))*1000000) + 1)

RNA_data = RNA_data.apply(Log_Norm,axis=0)
RNA_data_scaled = pd.DataFrame(data=st.zscore(RNA_data.T),index = RNA_data.columns,columns=RNA_data.index)

datadict = dict()
datadict['RNA_data'] = RNA_data.T
datadict['RNA_data_scaled'] = RNA_data_scaled

with open('data/Moffit_RNA.pkl','wb') as f:
    pickle.dump(datadict, f, protocol=4)


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
MERFISH_data_scaled = pd.DataFrame(data=st.zscore(MERFISH_data.T),index = MERFISH_data.columns,columns=MERFISH_data.index)

datadict = dict()
datadict['MERFISH_data'] = MERFISH_data.T
datadict['MERFISH_data_scaled'] = MERFISH_data_scaled
datadict['MERFISH_meta'] = MERFISH_meta

with open('data/MERFISH1.pkl','wb') as f:
    pickle.dump(datadict, f)

