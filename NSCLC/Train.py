#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.08.04               #
# **********************************#

import time
from Options import TrainOptions
from GAN_Model import GANModel
from Graph import graph_construction_scRNA, graph_construction_spatial
from Utils import sparse_mx_to_torch_sparse_tensor
import pandas as pd
import numpy as np
import scanpy as sc
import json
import torch

def neighbor_nodes(sparse_matrix, nodes_idx):
    return np.nonzero(sparse_matrix[nodes_idx].sum(axis=0))[1]

def retrieve_subgraph(graph_dict, expression_tensor, label_tensor, nodes_idx):
    subgraph_dict = {}
    subgraph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_norm'][nodes_idx,:][:,nodes_idx])
    subgraph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'][nodes_idx,:][:,nodes_idx])
    subgraph_dict['norm_value'] = graph_dict['norm_value']
    sub_expression_tensor = expression_tensor[nodes_idx]
    sub_label_tensor = label_tensor[nodes_idx]
    return subgraph_dict, sub_expression_tensor, sub_label_tensor

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


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()

    # create dataset.
    sc_adata = sc.read(opt.sc_data)
    st_adata = sc.read(opt.st_data)

    # Preparation
    raw_shared_gene = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    print(raw_shared_gene.shape, raw_shared_gene)

    #only use genes in both datasets
    common_st_adata = st_adata[:, raw_shared_gene].copy()
    common_sc_adata = sc_adata[:, raw_shared_gene].copy()
    graph_dict_sc = graph_construction_scRNA(common_sc_adata, n_neighbors=opt.sc_neighbors)
    x_sc = torch.Tensor(common_sc_adata.X)
    l_sc = torch.Tensor(common_sc_adata.obs.labels)

    types_dict = dict(set(zip(common_sc_adata.obs.labels.to_list(), common_sc_adata.obs.cell_types.to_list())))
    types_dict2 = dict(set(zip(common_sc_adata.obs.cell_types.to_list(), common_sc_adata.obs.labels.to_list())))

    graph_dict_st = graph_construction_spatial(common_st_adata, dis_sigma=opt.dis_sigma, n_neighbors=opt.st_neighbors)
    x_st = torch.Tensor(common_st_adata.X)
    l_st = torch.Tensor([0] * common_st_adata.shape[0])
    tensor_graph_dict_st = to_tensor(graph_dict_st)


    sc_cells_number, st_cells_number = common_sc_adata.shape[0], common_st_adata.shape[0]
    print('==> The number of cells in sc dataset: %d | in st dataset: %d' % (sc_cells_number, st_cells_number))
    sc_batches = int(sc_cells_number / opt.cells_per_batch)
    st_batches = int(st_cells_number / opt.cells_per_batch)
    batches = max(sc_batches, st_batches)
    print('==> Epochs: %d, Batches: %d, sc_batches: %d, st_batches: %d' %(opt.n_epochs, batches, sc_batches, st_batches))

    best = 20
    acc = 0

    tar_cell_type = ['Tumors 5', 'Tumors 4', 'Tumors 3', 'Tumors 2', 'Tumors 1', 'tumor endothelial', 'follicular B', 'MALT B', 
                     'COL4A2 fibroblasts', 'PLA2G2A fibroblasts', 'GABARAP fibroblasts', 'respiratory epithelial']
    cal_cell_type = ['tumors', 'tumors', 'tumors', 'tumors', 'tumors', 'endothelial', 'B-cell', 'B-cell', 'fibroblast', 'fibroblast', 'fibroblast', 'epithelial']

    setup_seed(42)
    model = GANModel(opt)
    model.setup(opt)        # regular setup: loading models for test or continue train, and print networks

    ### Training.
    for epoch in range(1, opt.n_epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>
        epoch_start_time = time.time()          # timer for entire epoch

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
            subgraph_dict_sc, subx_sc, subl_sc = retrieve_subgraph(graph_dict_sc, x_sc, l_sc, scb_subgraph_cells_idx)
            subgraph_dict_st, subx_st, subl_st = retrieve_subgraph(graph_dict_st, x_st, l_st, stb_subgraph_cells_idx)

            model.optimize_parameters(subx_sc, subl_sc, subgraph_dict_sc, subx_st, subgraph_dict_st) # calculate loss functions, get gradients, update network weights

            if batch % opt.print_freq == 0:
                losses = model.get_current_losses()  # ['CLS', 'GAN', 'REC', 'DIS']
                print('==> Batch:[%4d] | subgraphs: %d, %d | Loss: cls %.4f, gan %.4f, rec %.4f, dis %.4f'%(batch, 
                           len(scb_subgraph_cells_idx), len(stb_subgraph_cells_idx), losses['CLS'], losses['GAN'], losses['REC'], losses['DIS']))

        epoch_loss_cls = model.get_current_losses()['CLS']

        model.update_learning_rate()  # update learning rates based on epoch.
        print('===> End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        if epoch > 30:
            print('\n===> Start inferencing at epoch %d... | cls_loss %.4f'%(epoch, epoch_loss_cls))
            best = epoch_loss_cls
            ### Inferencing.
            results = model.inference(x_st, tensor_graph_dict_st).cpu().numpy()

            predictions = [int(i) for i in np.argmax(results, axis=1)]
            tmp_adata = common_st_adata.copy()
            tmp_adata.obs['pred_cls'] = [types_dict[i] for i in predictions]
            #tmp_adata = tmp_adata[~tmp_adata.obs.celltype_refined.isin(['monocyte']), :]
            arr1 = np.array(tmp_adata.obs['pred_cls'].replace(tar_cell_type, cal_cell_type))
            arr2 = np.array(tmp_adata.obs['celltype_refined'])
            best_acc = np.sum(arr1 == arr2) *1. / tmp_adata.shape[0]

            if best_acc > acc:
                acc = best_acc

                common_st_adata.obs['conf'] = np.max(results, axis=1)
                common_st_adata.obsm['results'] = results
                common_st_adata.obs['pred'] = predictions
                common_st_adata.obs['pred_cls'] = [types_dict[i] for i in predictions]

                print('\t### saving the model at the epoch %d with best acc %.4f'%(epoch, acc))
                model.save_networks(epoch)

    #result_df = pd.DataFrame(results, columns=[types_dict[i] for i in range(22)], index=common_st_adata.obs_names.to_list())
    #result_df.to_csv('Results/best_spagda_prediction.csv')

    print('Prediction distribution:', common_st_adata.obs.pred_cls.value_counts())
    print('best acc: %.4f'%acc)
    common_st_adata.write('Results/nsclc_pred_%d_%d_%d_%s.h5ad'%(opt.classes, opt.dis_sigma, opt.lambda_C, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
