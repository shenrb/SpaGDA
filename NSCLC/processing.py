#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   shen_rongbo@gzlab.ac.cn  #
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

def alignment(array):
    minx, maxx, miny, maxy = np.min(array[:,0]), np.max(array[:,0]), np.min(array[:,1]), np.max(array[:,1])
    return array - np.array([minx, miny]), maxx-minx, maxy-miny

def verizontal(array, height=0):
    horizonx = array[:,0]
    horizony = height - array[:,1]
    return np.stack((horizonx, horizony),axis=1)

if __name__ == '__main__':
    # Set arguments and hyper-parameters.
    parser = argparse.ArgumentParser(description='preprocess nsclc dataset for annotation.')
    parser.add_argument('--dataset', default='/data/bioinf/jyuan/rongboshen/datasets/nsclc', type=str, help='the experimental datasets.')
    parser.add_argument('--save_dir', default='data', type=str, help='the save dir.')
    args = parser.parse_args()

    sc_adata = sc.read(os.path.join(args.dataset, 'post_scRNA.h5ad'))
    sc.pp.filter_cells(sc_adata, min_counts=10)
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

    st_adata.write(os.path.join(args.save_dir, 'nsclc_ori.h5ad'))

    shared_genes = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    print(shared_genes.shape)
    print('missed genes in scRNA, but in st data:', set(st_adata.var_names) - set(shared_genes))

    with open(os.path.join(args.save_dir, 'shared_genes.json'), 'w') as fs:
        json.dump(list(shared_genes), fs)


    # normalized
    #sc.pp.normalize_total(sc_adata)
    #sc.pp.normalize_total(st_adata)

    ## filtered by shared genes
    sc_adata = sc_adata[:, shared_genes]
    st_adata = st_adata[:, shared_genes]

    del_types = ['lower quality endothelial cell', 'lower quality fibroblasts', 'lower quality epithelial cell']
    sc_adata = sc_adata[~sc_adata.obs.ClusterName.isin(del_types), :]




    src_cell_type = ['CD8+ T cells', 'CD4+ T cells', 'cancer cells pt 5', 'cancer cells pt 4', 'cancer cells pt 3', 'cancer cells pt 2', 'cancer cells pt 1',
                     'follicular B cells', 'natural killer cells', 'plasma B cells', 'MALT B cells', 'regulatory T cells', 'tumour endothelial cell', 'mast cells',
                     'normal endothelial cell', 'monocyte-derived dendritic cells', 'COL4A2-expressing fibroblasts', 'PLA2G2A-expressing fibroblasts', 
                     'GABARAP-expressing fibroblasts', 'respiratory epithelial cells', 'epithelial cell', 'plasmacytoid dendritic cells', 'granulocytes', 'macrophages']
    tar_cell_type = ['T CD8', 'T CD4', 'Tumors 5', 'Tumors 4', 'Tumors 3', 'Tumors 2', 'Tumors 1', 'follicular B', 'NK', 'plasmablast', 'MALT B',
                     'Treg', 'tumor endothelial', 'mast', 'endothelial', 'mDC', 'COL4A2 fibroblasts', 'PLA2G2A fibroblasts', 'GABARAP fibroblasts', 'respiratory epithelial',
                     'epithelial', 'pDC', 'granulocytes', 'macrophage']
    tar_cell_type2 = ['T CD8', 'T CD4', 'tumors', 'tumors', 'tumors', 'tumors', 'tumors', 'B-cell', 'NK', 'plasmablast', 'B-cell',
                      'Treg', 'endothelial', 'mast', 'endothelial', 'mDC', 'fibroblast', 'fibroblast', 'fibroblast', 'epithelial', 
                      'epithelial', 'pDC', 'neutrophil', 'macrophage']


    sc_adata.obs['cell_types'] = list(sc_adata.obs['ClusterName'].replace(src_cell_type, tar_cell_type2))

    class_names = list(set(sc_adata.obs['cell_types']))
    class_names.sort()
    print('%d class names'%(len(class_names)), class_names)
    name2num_dict = dict(zip(class_names, range(len(class_names))))

    # upsample rare classes
    upsample_types1 = ['pDC', 'epithelial', 'mDC']
    upsample_types2 = ['fibroblast', 'mast', 'Treg', 'neutrophil', 'endothelial'] #'COL4A2 fibroblasts', 'PLA2G2A fibroblasts', 'GABARAP fibroblasts', 'fibroblast', 
    downsamples = ['T CD8', 'T CD4', 'macrophage']
    adata_upsample1 = sc_adata[sc_adata.obs['cell_types'].isin(upsample_types1),:]
    adata_upsample2 = sc_adata[sc_adata.obs['cell_types'].isin(upsample_types2),:]
    adata_downsample = sc_adata[sc_adata.obs['cell_types'].isin(downsamples),:]
    adata_normal = sc_adata[~sc_adata.obs['cell_types'].isin(upsample_types1+upsample_types2+downsamples),:]

    downsample_ids = adata_downsample.obs.sample(frac=0.6).index.to_list()
    adata_downsample = adata_downsample[adata_downsample.obs.index.isin(downsample_ids), :]
    features_downsample = np.array(adata_downsample.X, dtype=np.float32)
    labels_downsample = adata_downsample.obs['cell_types'].to_list()

    features_upsample1 = np.array(adata_upsample1.X, dtype=np.float32)
    labels_upsample1 = adata_upsample1.obs['cell_types'].to_list()

    features_upsample2 = np.array(adata_upsample2.X, dtype=np.float32)
    labels_upsample2 = adata_upsample2.obs['cell_types'].to_list()

    features_normal = np.array(adata_normal.X, dtype=np.float32)
    labels_normal = adata_normal.obs['cell_types'].to_list()

    features_up1 = np.tile(features_upsample1, (10,1))
    shape = features_up1.shape
    noise = np.random.rand(shape[0],shape[1])
    features_up1 = features_up1 + 0.01*noise.astype(np.float32)*np.mean(features_up1)
    labels_up1 = labels_upsample1*10

    features_up2 = np.tile(features_upsample2, (2,1))
    shape = features_up2.shape
    noise = np.random.rand(shape[0],shape[1])
    features_up2 = features_up2 + 0.01*noise.astype(np.float32)*np.mean(features_up2)
    labels_up2 = labels_upsample2*2

    features = np.concatenate((features_normal,features_up1), axis=0)
    features = np.concatenate((features,features_up2), axis=0)
    features = np.concatenate((features,features_downsample), axis=0)    
    for i in range(features.shape[0]):
        norm = np.linalg.norm(features[i])
        if norm != 0:
            features[i] = features[i]/norm
    subclass_labels = labels_normal + labels_up1 + labels_up2 + labels_downsample

    obs = pd.DataFrame(index=range(features.shape[0]))
    var = pd.DataFrame(index=sc_adata.var_names.to_list())

    post_sc_adata = AnnData(features, obs=obs, var=var, dtype='float32')
    post_sc_adata.obs['cell_types'] = subclass_labels
    post_sc_adata.obs['labels'] = [name2num_dict[i] for i in subclass_labels]
    post_sc_adata.write(os.path.join(args.save_dir, 'scRNA2.h5ad'))


    #st_features = np.array(st_adata.X)
    #for i in range(st_features.shape[0]):
    #    norm = np.linalg.norm(st_features[i])
    #    if norm != 0:
    #        st_features[i] = st_features[i]/norm
    #st_adata.X = st_features
    #st_adata.write(os.path.join(args.save_dir, 'nsclc.h5ad'))



    # figure 5 need resize
    # figure 16 need resize y axis
    # merge figures
    files = os.listdir(os.path.join(args.dataset, 'figures'))
    files.sort()


    # all image crop zero edges and resized to (5472, 3650)

    for j,file in enumerate(files):
        img = cv2.imread(os.path.join(args.dataset, 'figures', file)) # height * width

        img_arr = np.sum(img, axis=2)
        f,g = np.nonzero(np.sum(img_arr, axis=0)), np.nonzero(np.sum(img_arr, axis=1))
        minx, maxx = np.min(f), np.max(f)
        miny, maxy = np.min(g), np.max(g)
        img = img[miny:maxy+1, minx:maxx+1, :]
        img = cv2.resize(img, dsize=(5472, 3650), interpolation=cv2.INTER_CUBIC) # dsize = resized_width * resized_height

        if j%4 == 0:
            tmp = img
        else:
            tmp = cv2.hconcat([tmp, img])

        if j == 3:
            merged_image = tmp

        if j%4 == 3 and j > 3:
            
            merged_image = cv2.vconcat([tmp, merged_image])

    print(merged_image.shape)
    cv2.imwrite(os.path.join(args.save_dir, 'merged_image.jpg'), merged_image)




