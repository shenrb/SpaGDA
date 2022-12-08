# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   # rongboshen2019@gmail.com
# Date:    2022.07.06               #
# ***********************************

import torch
from torch import nn, einsum
import torch.nn.functional as F

class Basic_MLP(nn.Module):
    def __init__(self, opt):
        super(Basic_MLP, self).__init__()

        input_dim = opt.gcn_hidden2 + opt.feat_hidden2
        hidden_dim = 30
        dropout = 0.5
        output_dim = opt.classes 

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

