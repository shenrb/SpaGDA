#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com # 
# Date:    2022.07.06               #
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

    graph_dict_st = graph_construction_spatial(common_st_adata, dis_sigma=opt.dis_sigma, n_neighbors=opt.st_neighbors)
    x_st = torch.Tensor(common_st_adata.X)
    l_st = torch.Tensor(common_st_adata.obs.labels)
    tensor_graph_dict_st = to_tensor(graph_dict_st)

    sc_cells_number, st_cells_number = common_sc_adata.shape[0], common_st_adata.shape[0]
    print('==> The number of cells in sc dataset: %d | in st dataset: %d' % (sc_cells_number, st_cells_number))
    sc_batches = int(sc_cells_number / opt.cells_per_batch)
    st_batches = int(st_cells_number / opt.cells_per_batch)
    batches = max(sc_batches, st_batches)
    print('==> Epochs: %d, Batches: %d, sc_batches: %d, st_batches: %d' %(opt.n_epochs, batches, sc_batches, st_batches))

    best = 0
    setup_seed(1) # Warnning: different random seed may generate slight different results.
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

        model.update_learning_rate()  # update learning rates based on epoch.
        print('===> End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        if epoch > 35: # save the best model from the latest five epoch, since the training should converge in the latest epoches and the results are about the same.
            print('\n===> Start inferencing at epoch %d...'%epoch)
            ### Inferencing.
            results = model.inference(x_st, tensor_graph_dict_st).cpu().numpy()
            predictions = [int(i) for i in np.argmax(results, axis=1)]
            gt_arr = np.array(common_st_adata.obs.labels)

            acc = sum(gt_arr==np.array(predictions))*1./common_st_adata.shape[0]
            print("Acc for prediction:", acc)

            if acc > best:
                best =  acc
                print('\t### saving the model at the epoch %d'%epoch)
                model.save_networks(epoch)
                best_results = results

    print("Best acc for prediction:", best)
    with open(opt.log_file, 'a') as opt_file:
        opt_file.write("Acc for prediction: %.4f"%best)

    result_df = pd.DataFrame(best_results, columns=[types_dict[i] for i in range(9)], index=common_st_adata.obs_names.to_list())
    result_df.to_csv('Results/best_spagda_prediction_%s.csv'%(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
