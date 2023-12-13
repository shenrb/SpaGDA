# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.17               #
# ***********************************

library(Seurat)
library(ggplot2)
library(dplyr)
library(magrittr)
library(Seurat)
library(SeuratDisk)
library(patchwork)
library(MLmetrics)
library(SingleR)

query_data <- readRDS("MERFISH.rds")
ref_data <- readRDS("Moffit_RNA.rds")
Gene_groups <- fromJSON(file = "gene_groups.json")

genes.leaveout <- intersect(rownames(query_data),rownames(ref_data))

query_data <- ScaleData(query_data, features = genes.leaveout)
ref_data <- ScaleData(ref_data, features = genes.leaveout)

acc <- c()
f1 <- c()
f1w <- c()

## Predict lable from the reference data
ref_matrix <- as.matrix(GetAssayData(object = ref_data, slot = "data"))
query_matrix <- as.matrix(GetAssayData(object = query_data, slot = "data"))
result <- SingleR(test=query_matrix, ref=ref_matrix, labels=ref_data$celltype)

y_true <- as.character(query_data$celltype)
y_pred <- as.character(result$labels)
print(table(y_true, y_pred))
acc <- append(acc, Accuracy(y_true, y_pred))
f1 <- append(f1, F1_Score_macro(y_true, y_pred))
f1w <- append(f1w, F1_Score_macro_weighted(y_true, y_pred))

print(acc)
print(f1)
print(f1w)
