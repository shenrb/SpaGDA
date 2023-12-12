#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2023.04.18               #
# ***********************************

import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.io as io
import scanpy as sc
from anndata import AnnData
import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import argparse
from scipy import stats
import seaborn as sns
import anndata as ad
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import random

def alignment(array):
    minx, maxx, miny, maxy = np.min(array[:,0]), np.max(array[:,0]), np.min(array[:,1]), np.max(array[:,1])
    return array - np.array([minx, miny]), maxx-minx, maxy-miny

def verizontal(array, height=0):
    horizonx = array[:,0]
    horizony = height - array[:,1]
    return np.stack((horizonx, horizony),axis=1)



def filter_outliers(adata, expr, percent=0.2, sample=None):
    #cos_sims = cosine_similarity(expr.reshape(1, -1), np.exp(adata.X.toarray())-1)
    cos_sims = cosine_similarity(expr.reshape(1, -1), adata.layers['count'].toarray())
    rems = int(adata.shape[0] * (1-percent))
    rems_index = cos_sims.argsort().flatten()[-rems:]

    if sample:
        random.shuffle(rems_index)
        rems_index = rems_index[:int(len(rems_index)*sample)]

    indx = [False]*adata.shape[0]
    for ri in rems_index:
        indx[ri] = True

    return adata[indx, :]


def filter_differs(adata, expr, percent=0.2):
    cos_sims = cosine_similarity(expr.reshape(1, -1), adata.X)
    rems = int(adata.shape[0] * (1-percent))
    rems_index = cos_sims.argsort().flatten()[:rems]
    indx = [False]*adata.shape[0]
    for ri in rems_index:
        indx[ri] = True

    return adata[indx, :]
   


if __name__ == '__main__':
    # Set arguments and hyper-parameters.
    parser = argparse.ArgumentParser(description='preprocess nsclc dataset for annotation.')
    parser.add_argument('--dataset', default='/data/bioinf/jyuan/rongboshen/datasets/nsclc', type=str, help='the experimental datasets.')
    parser.add_argument('--save_dir', default='data', type=str, help='the save dir.')
    args = parser.parse_args()

    sc_adata = sc.read(os.path.join(args.dataset, 'local.h5ad'))
    sc.pp.filter_genes(sc_adata, min_cells=1)
    print('scRNA shape:', sc_adata.shape)


    st_adata = sc.read(os.path.join(args.dataset, 'nanostring_fov1_sampledata.h5ad'))
    coord = st_adata.obsm['spatial']
    _coord, xrang, yrang = alignment(verizontal(coord))
    _coord[:,0] = _coord[:,0]*5462.0/xrang + 5
    _coord[:,1] = _coord[:,1]*3640.0/yrang + 5
    st_adata.obsm['spatial'] = _coord.astype(np.int32)

    for i in range(2,21):
        tdata = sc.read(os.path.join(args.dataset, 'nanostring_fov%d_sampledata.h5ad')%i)
        coord = tdata.obsm['spatial']
        _coord, xrang, yrang = alignment(verizontal(coord))
        _coord[:,0] = _coord[:,0]*5462.0/xrang + 5
        _coord[:,1] = _coord[:,1]*3640.0/yrang + 5
        m,n = int(i-1)%4, int((i-1)/4)
        tdata.obsm['spatial'] = _coord.astype(np.int32) + np.array([5472*m,3650*n], dtype=np.int32)

        st_adata = ad.concat([st_adata, tdata], join='outer', fill_value=0)

    st_adata.obsm['spatial'] = verizontal(st_adata.obsm['spatial'], height=18250)

    num_cells = st_adata.shape[0]
    st_adata.obs['NUM'] = [str(i) for i in range(num_cells)]
    st_adata.obs.set_index('NUM', inplace=True)

    #st_adata.write(os.path.join(args.save_dir, 'nsclc_ori.h5ad'))
    sc_adata.var.set_index('feature_name', inplace=True)
    shared_genes = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    print(shared_genes.shape)
    print('missed genes in scRNA, but in st data:', set(st_adata.var_names) - set(shared_genes))

    with open(os.path.join(args.save_dir, 'shared_genes_cc.json'), 'w') as fs:
        json.dump(list(shared_genes), fs)

    sc.settings.figdir = args.save_dir

    ## filtered by shared genes
    sc_adata = sc_adata[:, shared_genes]
    sc.pp.filter_cells(sc_adata, min_counts=10)
    st_adata = st_adata[:, shared_genes]

    del_types = ['type II pneumocyte', 'type I pneumocyte', 'myeloid cell', 'pericyte', 'dendritic cell', 
                 'conventional dendritic cell', 'smooth muscle cell', 'club cell', 'mesothelial cell', 'stromal cell']
    sc_adata = sc_adata[~sc_adata.obs['cell_type'].isin(del_types), :]
    sc_adata = sc_adata[sc_adata.obs['tissue'].isin(['lung']), :]
    print('scRNA dataset shape:', sc_adata.shape)
    print('scRNA dataset cell type distribution:\n', sc_adata.obs['cell_type'].value_counts())


    src_cell_type = ['alveolar macrophage', 'CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell', 'malignant cell', 'classical monocyte', 
                     'natural killer cell', 'B cell', 'epithelial cell of lung', 'regulatory T cell', 'CD1c-positive myeloid dendritic cell', 'plasma cell',
                     'vein endothelial cell', 'capillary endothelial cell', 'mast cell', 'multi-ciliated epithelial cell', 'fibroblast of lung', 
                     'endothelial cell of lymphatic vessel', 'bronchus fibroblast of lung', 'non-classical monocyte', 'pulmonary artery endothelial cell', 
                     'plasmacytoid dendritic cell']

    tar_cell_type = ['macrophage', 'T CD4', 'T CD8', 'tumors', 'monocyte', 'NK', 'B-cell', 'epithelial', 'Treg', 'mDC', 'plasmablast', 'endothelial', 'endothelial',
                     'mast', 'epithelial', 'fibroblast', 'endothelial', 'fibroblast', 'monocyte', 'endothelial', 'pDC']




    sc_adata.obs['cell_types'] = list(sc_adata.obs['cell_type'].replace(src_cell_type, tar_cell_type))
    print('scRNA dataset refined cell type distribution: \n', sc_adata.obs['cell_types'].value_counts())


    class_names = list(set(sc_adata.obs['cell_types']))
    class_names.sort()
    print('%d class names'%(len(class_names)), class_names)
    name2num_dict = dict(zip(class_names, range(len(class_names))))



    # filter outliers and merge
    mean_expr_dict = {}
    for i, ct in enumerate(class_names):
        sub_adata = sc_adata[sc_adata.obs.cell_types==ct,:]
        mean_expr = np.mean(sub_adata.layers['count'].toarray(), axis=0)
        mean_expr_dict[ct] = mean_expr


    # upsample rare classes
    upsample_types1 = ['pDC']
    upsample_types2 = ['fibroblast', 'mast', 'neutrophil'] 
    downsamples = ['T CD8', 'T CD4', 'macrophage']

    center_sample_cts = ['macrophage', 'T CD8', 'Treg', 'plasmablast', 'endothelial', 'neutrophil', 'mast', 
                         'fibroblast', 'pDC', 'B-cell', 'T CD4', 'NK', 'tumors', 'monocyte', 'epithelial', 'mDC']

    for i, ct in enumerate(center_sample_cts):
        sub_adata = sc_adata[sc_adata.obs.cell_types==ct,:]
        if ct == 'macrophage':
            sub_adata = filter_outliers(sub_adata, mean_expr_dict[ct], 0.5, 0.4)
        elif ct == 'T CD8' or ct == 'T CD4' or ct == 'monocyte':
            sub_adata = filter_outliers(sub_adata, mean_expr_dict[ct], 0.5, 0.7)
        elif ct in upsample_types2 + upsample_types1:
            sub_adata = filter_outliers(sub_adata, mean_expr_dict[ct], 0.1)
        else:
            sub_adata = filter_outliers(sub_adata, mean_expr_dict[ct], 0.5)

        if i == 0:
            fsc_adata = sub_adata
        else:
            fsc_adata = ad.concat([fsc_adata, sub_adata], join='outer', fill_value=0)

    # draw umap of fsc_adata

    fsc_adata2 = fsc_adata.copy()
    fsc_adata2.X = fsc_adata2.layers['count']
    sc.pp.normalize_total(fsc_adata2)
    sc.pp.pca(fsc_adata2)
    sc.pp.neighbors(fsc_adata2)
    sc.pl.umap(fsc_adata2, color='cell_types', legend_fontsize='x-small', save='_umap1.pdf')
    sc.tl.umap(fsc_adata2)
    sc.pl.umap(fsc_adata2, color='cell_types', legend_fontsize='x-small', save='_umap2.pdf')


    # upsample for pDC and neutrophil
    adata_upsample1 = fsc_adata[fsc_adata.obs['cell_types']=='pDC',:]
    adata_upsample2 = fsc_adata[fsc_adata.obs['cell_types']=='neutrophil',:]
    adata_normal = fsc_adata[~fsc_adata.obs['cell_types'].isin(['pDC', 'neutrophil']),:]

    features_upsample1 = np.array(adata_upsample1.layers['count'].toarray(), dtype=np.float32)
    features_upsample2 = np.array(adata_upsample2.layers['count'].toarray(), dtype=np.float32)

    labels_upsample1 = adata_upsample1.obs['cell_types'].to_list()
    labels_upsample2 = adata_upsample2.obs['cell_types'].to_list()

    features_normal = np.array(adata_normal.layers['count'].toarray(), dtype=np.float32)
    labels_normal = adata_normal.obs['cell_types'].to_list()

    features_up1 = np.tile(features_upsample1, (5,1))
    shape = features_up1.shape
    noise = np.random.rand(shape[0],shape[1])
    features_up11 = features_up1 + 0.01*noise.astype(np.float32)*np.mean(features_up1)
    labels_up11 = labels_upsample1*6

    features_up2 = features_upsample2
    shape = features_up2.shape
    noise = np.random.rand(shape[0],shape[1])
    features_up22 = features_up2 + 0.01*noise.astype(np.float32)*np.mean(features_up2)
    labels_up22 = labels_upsample2*2


    features = np.concatenate((features_normal,features_upsample1), axis=0)  
    features = np.concatenate((features,features_up11), axis=0) 
    features = np.concatenate((features,features_upsample2), axis=0) 
    features = np.concatenate((features,features_up22), axis=0) 

    for i in range(features.shape[0]):
        norm = np.linalg.norm(features[i])
        if norm != 0:
            features[i] = features[i]/norm
    subclass_labels = labels_normal + labels_up11 + labels_up22

    obs = pd.DataFrame(index=range(features.shape[0]))
    var = pd.DataFrame(index=fsc_adata.var_names.to_list())

    post_sc_adata = AnnData(features, obs=obs, var=var, dtype='float32')
    post_sc_adata.obs['cell_types'] = subclass_labels
    post_sc_adata.obs['labels'] = [name2num_dict[i] for i in subclass_labels]
    post_sc_adata.write(os.path.join(args.save_dir, 'scRNA_cc5.h5ad'))  # scRNA.h5ad not umsample endothelial

    print('post scRNA dataset shape: \n', post_sc_adata.shape)
    print('post scRNA dataset refined cell type distribution: \n', post_sc_adata.obs['cell_types'].value_counts())


