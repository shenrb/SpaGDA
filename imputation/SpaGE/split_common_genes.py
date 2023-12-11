#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.05.09               #
# ***********************************

import os
import pickle
import numpy as np
import pandas as pd
import json

with open ('SpaGE_pkl/MERFISH1.pkl', 'rb') as f:
    datadict = pickle.load(f)

MERFISH_data = datadict['MERFISH_data']
MERFISH_data_scaled = datadict['MERFISH_data_scaled']
MERFISH_meta = datadict['MERFISH_meta']
del datadict

with open ('SpaGE_pkl/Moffit_RNA.pkl', 'rb') as f:
    datadict = pickle.load(f)
    
RNA_data = datadict['RNA_data']
RNA_data_scaled = datadict['RNA_data_scaled']
del datadict

raw_shared_gene = np.intersect1d(MERFISH_data_scaled.columns, RNA_data_scaled.columns)
print(raw_shared_gene.shape)
print(raw_shared_gene)

#### Split into 5 Folds for Cross Validation ####
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

folds = {}
number = 1
for train_index, test_index in kf.split(raw_shared_gene):
    folds[number] = list(raw_shared_gene[test_index])
    number += 1
    print(raw_shared_gene[test_index])

with open('../gene_groups.json', 'w') as fs:
    json.dump(folds, fs)


