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

SEED = 2021
data = sc.read_h5ad('/aaa/jianhuayao/project2/data/Zheng68k/Zheng68k_full.h5ad')
sc.pp.filter_cells(data, min_genes=200)
sc.pp.normalize_total(data, target_sum=1e6)
sc.pp.log1p(data, base=2)
acc = []
f1 = []
f1w = []
for rep in range(5):
    SEED += 1
    # data_train, data_val = train_test_split(data, test_size=0.1,random_state=SEED)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
    for index_train, index_val, *_ in sss.split(data, data.obs['celltype']):
        data_train = data[index_train]
        data_val = data[index_val]
        ## preprocess training data
        sc.pp.highly_variable_genes(data_train, n_top_genes=3000)
        sc.pp.pca(data_train)
        sc.pp.neighbors(data_train)
        ## preprocess test data
        sc.pp.highly_variable_genes(data_val, n_top_genes=3000)
        sc.pp.pca(data_val)
        sc.pp.neighbors(data_val)
        data_val.obs["true_celltype"] = data_val.obs["celltype"]
        data_val.obs["celltype"] = "Unlabeled"
        adata = data_train.concatenate(data_val)
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
        data_val.obs['scNym'] = np.array(adata.obs.loc[[x + '-1' for x in data_val.obs_names], 'scNym'])
        y_true = data_val.obs['true_celltype']
        y_pred = data_val.obs['scNym']
        acc.append(accuracy_score(y_true, y_pred))
        f1.append(f1_score(y_true, y_pred, average='macro'))
        f1w.append(f1_score(y_true, y_pred, average='weighted'))
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4))
    print(acc)
    print(f1)
    print(f1w)

print(acc)
print(f1)
print(f1w)