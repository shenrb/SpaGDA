#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.06.15               #
# ***********************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NLayerDiscriminator(nn.Module):
    """Defines a GAN discriminator"""

    def __init__(self, ndf=4, n_layers=3, kw=4, stride=2, norm_layer=nn.BatchNorm2d):
        """Construct a GAN discriminator

        Parameters:
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        padw = 1
        sequence = [nn.Conv2d(1, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        input = torch.unsqueeze(torch.unsqueeze(input, 0), 0)
        return self.model(input)

class NLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(NLayerMLP, self).__init__()

        hidden_dim1 = 30
        hidden_dim2 = 10

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.net(x)
