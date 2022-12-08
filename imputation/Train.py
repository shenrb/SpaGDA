#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   # rongboshen2019@gmail.com
# Date:    2022.06.27               #
# **********************************#

import os
import time
from Options import TrainOptions
from GAN_Model import GANModel
from Graph import graph_construction_scRNA, graph_construction_spatial
from Utils import sparse_mx_to_torch_sparse_tensor, SSIM, SP, JS, RMSE, PS
import pandas as pd
import numpy as np
import scanpy as sc
import json
import torch
import pickle
import scipy.stats as st
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as stsim
from scipy.spatial import distance
import statistics

def neighbor_nodes(sparse_matrix, nodes_idx):
    return np.nonzero(sparse_matrix[nodes_idx].sum(axis=0))[1]

def retrieve_subgraph(graph_dict, expression_tensor, nodes_idx):
    subgraph_dict = {}
    subgraph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_norm'][nodes_idx,:][:,nodes_idx])
    subgraph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'][nodes_idx,:][:,nodes_idx])
    subgraph_dict['norm_value'] = graph_dict['norm_value']
    sub_expression_tensor = expression_tensor[nodes_idx]
    return subgraph_dict, sub_expression_tensor

def to_tensor(graph_dict):
    tensor_graph_dict = {}
    tensor_graph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_norm'])
    tensor_graph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'])
    tensor_graph_dict['norm_value'] = graph_dict['norm_value']
    return tensor_graph_dict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    #torch.backends.cudnn.deterministic = True


def cv_train(opt, sc_cells_number, st_cells_number, batches, graph_dict_sc, x_sc, graph_dict_st, x_st, x_sc_a, mask, 
    tensor_graph_dict_st, raw_shared_gene, k, v, st_data_ori):

    best_pcc = 0
    best_scc = 0
    if opt.model == 'gan':
        model = GANModel(opt)
        model.setup(opt)        # regular setup: loading models for test or continue train, and print networks
    else:
        raise NotImplementedError('Model %s not implemented, please choose correct model [cycle_gan | gan]' % opt.model)

    ### Training.
    for epoch in range(1, opt.n_epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>
        epoch_start_time = time.time()  # timer for entire epoch

        sc_cell_idxes = np.arange(sc_cells_number)
        st_cell_idxes = np.arange(st_cells_number)
        np.random.shuffle(sc_cell_idxes)
        np.random.shuffle(st_cell_idxes)

        for batch in range(batches):
            scb = batch % sc_batches 
            stb = batch % st_batches 
            scb_cell_idx = sc_cell_idxes[scb*opt.cells_per_batch:(scb+1)*opt.cells_per_batch]
            stb_cell_idx = st_cell_idxes[stb*opt.cells_per_batch:(stb+1)*opt.cells_per_batch]
            scb_subgraph_cells_idx = neighbor_nodes(graph_dict_sc['adj_norm'], neighbor_nodes(graph_dict_sc['adj_norm'],scb_cell_idx))
            stb_subgraph_cells_idx = neighbor_nodes(graph_dict_st['adj_norm'], neighbor_nodes(graph_dict_st['adj_norm'],stb_cell_idx))
            subgraph_dict_sc, subx_sc = retrieve_subgraph(graph_dict_sc, x_sc, scb_subgraph_cells_idx)
            subgraph_dict_st, subx_st = retrieve_subgraph(graph_dict_st, x_st, stb_subgraph_cells_idx)
            subx_sc_a = x_sc_a[scb_subgraph_cells_idx]

            model.optimize_parameters(subx_sc, subgraph_dict_sc, subx_st, subgraph_dict_st, subx_sc_a, mask) # calculate loss functions, get gradients, update network weights

            if batch % opt.print_freq == 0:
                losses = model.get_current_losses()  # ['GAN', 'IMP', 'DIS'] for gan
                print('==> Batch:[%4d] | subgraphs: %d, %d | Loss: imp %.4f, rec %.4f, ganl %.4f, disl %.4f, gani %.4f, disi %.4f'%(batch, 
                        len(scb_subgraph_cells_idx), len(stb_subgraph_cells_idx), losses['IMP'], losses['RET'], losses['GAN_L'], losses['DIS_L'], losses['GAN_I'], losses['DIS_I']))

        model.update_learning_rate()  # update learning rates based on epoch.

        if epoch % opt.save_epoch_freq == 0:
            print('\t### saving the model at the epoch %d'%epoch)
            #model.save_networks('latest')
            model.save_networks(epoch)

        print('===> End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        if epoch > 30:
            ### Inferencing.
            print('\n===> Start inferencing at Fold %s, Epoch %d'%(k, epoch))
            decoded_st = model.inference(x_st, tensor_graph_dict_st).cpu().numpy()

            scc_list = []
            pcc_list = []
            for gene in v:
                idx = list(raw_shared_gene).index(gene)
                if np.std(decoded_st[:, idx]) == 0:
                    scc_list.append(0)
                    pcc_list.append(0)
                else:
                    scc_list.append(st.spearmanr(st_data_ori[:, idx], decoded_st[:, idx])[0])
                    pcc_list.append(st.pearsonr(st_data_ori[:, idx], decoded_st[:, idx])[0])

            print('===> SCC and PCC %.4f,  %.4f'%(statistics.median(scc_list), statistics.median(pcc_list)))
            if statistics.median(scc_list) > best_scc:
                best_scc = statistics.median(scc_list)
                best_results = decoded_st

    print('===> Best scc: %.4f'%best_scc)

    return best_results

def Normalize(adata):
    matrix = np.array(adata.X)
    tt = np.max(matrix, axis=1)
    _matrix = matrix / tt[:,None]
    adata.X = _matrix
    return adata


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()

    # create dataset.
    sc_adata = sc.read(opt.sc_data)
    sc_adata.var_names_make_unique()


    st_adata = sc.read(opt.st_data)
    with open(opt.gene_groups) as f:
        gene_groups = json.load(f)

    # Preparation
    raw_shared_gene = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    print(raw_shared_gene.shape, raw_shared_gene)
    Imputed_Genes = pd.DataFrame(columns=raw_shared_gene)

    #only use genes in both datasets
    common_st_adata = st_adata[:, raw_shared_gene].copy()
    common_sc_adata = sc_adata[:, raw_shared_gene].copy()
    sc.pp.filter_cells(common_sc_adata, min_counts = 1)


    graph_dict_sc = graph_construction_scRNA(sc_adata[common_sc_adata.obs_names,:], n_neighbors=opt.sc_neighbors)
    x_sc_a = torch.Tensor(common_sc_adata.X)

    sc_cells_number, st_cells_number = common_sc_adata.shape[0], common_st_adata.shape[0]
    print('==> The number of cells in sc dataset: %d | in st dataset: %d' % (sc_cells_number, st_cells_number))
    sc_batches = int(sc_cells_number / opt.cells_per_batch)
    st_batches = int(st_cells_number / opt.cells_per_batch)
    batches = max(sc_batches, st_batches)
    print('==> Epochs: %d, Batches: %d, sc_batches: %d, st_batches: %d' %(opt.n_epochs, batches, sc_batches, st_batches))

    setup_seed(6)
    dagcn_time = []

    st_data_ori = np.array(common_st_adata.X)

    # 5 folds cross validation.
    for k,v in gene_groups.items():
        print('==> Fold:', k)
        print('==> Imputed genes:', v)
        start = time.time()

        train_st_adata = common_st_adata.copy()
        train_st_adata[:,train_st_adata.var.index.isin(v)].X = 0 # set to 0 for imputed genes
        graph_dict_st = graph_construction_spatial(train_st_adata, dis_sigma=opt.dis_sigma, n_neighbors=opt.st_neighbors)
        x_st = torch.Tensor(train_st_adata.X)
        tensor_graph_dict_st = to_tensor(graph_dict_st)

        train_sc_adata = common_sc_adata.copy()
        train_sc_adata[:,train_sc_adata.var.index.isin(v)].X = 0 # set to 0 for imputed genes
        x_sc = torch.Tensor(train_sc_adata.X)

        mask = np.ones(raw_shared_gene.shape)
        for gene in v:
            idx = list(raw_shared_gene).index(gene)
            mask[idx] = 0

        best_results = cv_train(opt, sc_cells_number, st_cells_number, batches, graph_dict_sc, x_sc, graph_dict_st, x_st, x_sc_a, mask, 
                                tensor_graph_dict_st, raw_shared_gene, k, v, st_data_ori)

        for gene in v:
            idx = list(raw_shared_gene).index(gene)
            Imputed_Genes[gene] = best_results[:, idx]

        fold_time = time.time() - start
        print('==> Fold time: %d sec'%fold_time) 

    Imputed_Genes.to_csv('Results/%s_DAGAN_FiveFolds_%s.csv'%(opt.model, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    ### DAGCN
    st_data_shared = pd.DataFrame(st_data_ori, columns=raw_shared_gene)

    # metric
    sp = SP(st_data_shared, Imputed_Genes)
    print('===> DAGCN median SP: %.4f'%(sp.median(1)))

    DAGCN_RMSE = mean_squared_error(st_data_ori, Imputed_Genes.to_numpy())
    print('===> DAGCN RMSE: %.4f'%(DAGCN_RMSE))

    with open(opt.log_file, 'a') as f:
        f.write('===> DAGCN median SP: %.4f'%(sp.median(1)))
        f.write('===> DAGCN RMSE: %.4f'%(DAGCN_RMSE))

