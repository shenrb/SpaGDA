#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.06.15               #
# ***********************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the label tensor that has the same size as the input.
    """

    def __init__(self, gan_mode='vanilla', source_label=1.0, target_label=0.0, device='cpu'):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            source_label (bool) - - label for a source sample
            target_label (bool) - - label for a target sample

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('source_label', torch.tensor(source_label))
        self.register_buffer('target_label', torch.tensor(target_label))
        self.gan_mode = gan_mode
        self.device = device
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_label_tensor(self, prediction, is_source):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            is_source (bool) - - if the ground truth label is for source sample or target sample

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if is_source:
            label_tensor = self.source_label
        else:
            label_tensor = self.target_label
        return label_tensor.expand_as(prediction)

    def __call__(self, prediction, is_source):
        """Calculate loss given Discriminator's output and source/target labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            is_source (bool) - - if the ground truth label is for source sample or target sample

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            label_tensor = self.get_label_tensor(prediction, is_source).to(self.device)
            loss = self.loss(prediction.to(self.device), label_tensor)
        elif self.gan_mode == 'wgangp':
            if is_source:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GCNLoss(nn.Module):
    """Define GCN Loss.

    The GCN Loss class used to assess input and output of VGAE.
    """

    def __init__(self, mask=None, device='cpu'):
        """ Initialize the GCNLoss class.
        """
        super(GCNLoss, self).__init__()
        self.mask = mask
        self.device = device


    def __call__(self, preds, labels, mu, logstd, n_nodes, norm):

        if self.mask is not None:
            preds = preds * self.mask
            labels = labels * self.mask

        labels = labels.to_dense()
        pos_numbers = torch.nonzero(labels).size()[0]
        subgraph_pos_weight = float(labels.shape[0] * labels.shape[0] - pos_numbers) / pos_numbers
        cost = norm * F.binary_cross_entropy_with_logits(preds.to(self.device), labels.to(self.device), 
                                                         pos_weight=torch.tensor(subgraph_pos_weight))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), 1))
        return cost + KLD


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
