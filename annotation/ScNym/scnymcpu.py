# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.17               #
# ***********************************

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scnym
import time
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report

ref_data = sc.read('.h5ad')
query_data = sc.read('.h5ad')
acc = []
f1 = []
f1w = []

## preprocess training data
sc.pp.pca(ref_data)
sc.pp.neighbors(ref_data)
## preprocess test data
sc.pp.pca(query_data)
sc.pp.neighbors(query_data)
query_data.obs["true_celltype"] = query_data.obs["celltype"]
query_data.obs["celltype"] = "Unlabeled"
adata = ref_data.concatenate(query_data)

## train
scnym.api.scnym_api(
            adata=adata,
            task="train",
            groupby="celltype",
            config="no_new_identity",
            out_path="./scnym_outputs",
        )
## predict
scnym.api.scnym_api(
            adata=adata,
            task='predict',
            trained_model='./scnym_outputs',
        )

query_data.obs['scNym'] = np.array(adata.obs.loc[[x + '-1' for x in query_data.obs_names], 'scNym'])
y_true = query_data.obs['true_celltype']
y_pred = query_data.obs['scNym']
acc.append(accuracy_score(y_true, y_pred))
f1.append(f1_score(y_true, y_pred, average='macro'))
f1w.append(f1_score(y_true, y_pred, average='weighted'))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))

print(acc)
print(f1)
print(f1w)
