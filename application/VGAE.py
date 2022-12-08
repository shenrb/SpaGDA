#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   # rongboshen2019@gmail.com
# Date:    2022.06.14               #
# ***********************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )

def full_block_bn(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )

def full_block_relu(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        #nn.Dropout(p=p_drop),
    )

def full_block_bn_relu(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )

def full_block_bn_gelu(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )

def full_connect(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        #nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        #nn.ReLU(),
        #nn.Dropout(p=p_drop),
    )

# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, training, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.training = training
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, training, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.training = training

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj 


class VGAEModel(nn.Module):
    def __init__(self, opt):
        super(VGAEModel, self).__init__()
        self.training = opt.isTrain
        self.latent_dim = opt.gcn_hidden2 + opt.feat_hidden2

        # Feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block_bn(opt.input_dim, opt.feat_hidden1, opt.p_drop))
        self.encoder.add_module('encoder_L2', full_block_bn(opt.feat_hidden1, opt.feat_hidden2, opt.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L1', full_block_bn(self.latent_dim, opt.feat_hidden1, opt.p_drop))
        self.decoder.add_module('decoder_L2', full_block_bn_relu(opt.feat_hidden1, opt.input_dim, 0))

        # GCN layers
        self.gc1 = GraphConvolution(opt.feat_hidden2, opt.gcn_hidden1, opt.isTrain, opt.p_drop, act=lambda x: x)  #F.relu, F.elu,
        self.gc2 = GraphConvolution(opt.gcn_hidden1, opt.gcn_hidden2, opt.isTrain, opt.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(opt.gcn_hidden1, opt.gcn_hidden2, opt.isTrain, opt.p_drop, act=lambda x: x)

        self.dc = InnerProductDecoder(opt.p_drop, training=opt.isTrain, act=lambda x: x)


    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd) 
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        gcn_mu, gcn_logstd, encoded_x = self.encode(x, adj)
        gcn_z = self.reparameterize(gcn_mu, gcn_logstd)
        z = torch.cat((encoded_x, gcn_z), 1)
        decoded_x = self.decoder(z)
        #decoded_adj = self.dc(z)
        return gcn_mu, gcn_logstd, encoded_x, gcn_z, z, decoded_x#, decoded_adj




