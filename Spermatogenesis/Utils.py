#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.06.15               #
# **********************************#

import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import functools
import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats as st

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=opt.gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def cal_ssim(im1, im2, M):
    """
        calculate the SSIM value between two arrays.       
    Parameters
        -------
        im1: array1, shape dimension = 2
        im2: array2, shape dimension = 2
        M: the max value in [im1, im2]        
    """
    
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return ssim


def scale_max(df): 
    """
        Divided by maximum value to scale the data between [0,1].
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb
           
        Parameters
        -------
        df: dataframe, each col is a feature.
    """
    
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.max()
        result = pd.concat([result, content],axis=1)
    return result

def scale_z_score(df):  
    """
        scale the data by Z-score to conform the data to the standard normal distribution, that is, the mean value is 0, the standard deviation is 1, and the conversion function is 0.
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb
                
        Parameters
        -------
        df: dataframe, each col is a feature.       
        """
    
    result = pd.DataFrame()
    for label, content in df.items():
        content = stats.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result

def scale_plus(df):  
    """
        Divided by the sum of the data to scale the data between (0,1), and the sum of data is 1.
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb
              
        Parameters
        -------
        df: dataframe, each col is a feature.       
    """
    
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.sum()
        result = pd.concat([result,content],axis=1)
    return result


def SSIM(raw, impute, scale = 'scale_max'):
        
    ###This was used for calculating the SSIM value between two arrays.
        
    if scale == 'scale_max':
        raw = scale_max(raw)
        impute = scale_max(impute)
    else:
        print ('Please note you do not scale data by max')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col =  raw.loc[:,label]
            impute_col = impute.loc[:,label]
                
            M = [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
            raw_col_2 = np.array(raw_col)
            raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0],1)
                
            impute_col_2 = np.array(impute_col)
            impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0],1)
                
            ssim_res = cal_ssim(raw_col_2,impute_col_2,M)
                
            ssim_df = pd.DataFrame(ssim_res, index=["SSIM"],columns=[label])
            result = pd.concat([result, ssim_df],axis=1)
        return result
    else:
        print("columns error")

def SP(raw, impute, scale = None):      
    ###This was used for calculating the Pearson value between two arrays.
      
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col =  raw.loc[:,label]
            impute_col = impute.loc[:,label]
            if np.std(impute_col) == 0:
                spearmanr = 0
            else:
                spearmanr, _ = st.spearmanr(raw_col,impute_col)
            spearmanr_df = pd.DataFrame(spearmanr, index=["Spearman"],columns=[label])
            result = pd.concat([result, spearmanr_df],axis=1)
        return result

def PS(raw, impute, scale = None):
    ###This was used for calculating the Pearson value between two arrays.

    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col =  raw.loc[:,label]
            impute_col = impute.loc[:,label]
            if np.std(impute_col) == 0:
                pearsonr = 0
            else:
                pearsonr, _ = st.pearsonr(raw_col,impute_col)
            pearson_df = pd.DataFrame(pearsonr, index=["Pearson"],columns=[label])
            result = pd.concat([result, pearson_df],axis=1)
        return result

        
def JS(raw, impute, scale = 'scale_plus'):
        
    ###This was used for calculating the JS value between two arrays.
        
    if scale == 'scale_plus':
        raw = scale_plus(raw)
        impute = scale_plus(impute)
    else:
        print ('Please note you do not scale data by plus')    
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = np.array(raw.loc[:,label])
            impute_col = np.array(impute.loc[:,label])
                
            M = (raw_col + impute_col)/2
            KL = 0.5*st.entropy(raw_col,M)+0.5*st.entropy(impute_col,M)
            KL_df = pd.DataFrame(KL, index=["JS"],columns=[label])
                
                
            result = pd.concat([result, KL_df],axis=1)
        return result

        
def RMSE(raw, impute, scale = 'zscore'):
        
    ###This was used for calculating the RMSE value between two arrays.
        
    if scale == 'zscore':
        raw = scale_z_score(raw)
        impute = scale_z_score(impute)
    else:
        print ('Please note you do not scale data by zscore')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col =  raw.loc[:,label]
            impute_col = impute.loc[:,label]
                
            RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())
            RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
                
            result = pd.concat([result, RMSE_df],axis=1)
        return result
