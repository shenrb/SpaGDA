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
library(SeuratDisk)
library(patchwork)
library(MLmetrics)
library(SingleCellExperiment)
library(scmap)

query_data <- readRDS("MERFISH.rds")
ref_data <- readRDS("Moffit_RNA.rds")
Gene_groups <- fromJSON(file = "gene_groups.json")

genes.leaveout <- intersect(rownames(query_data),rownames(ref_data))
query_data <- ScaleData(query_data, features = genes.leaveout)
ref_data <- ScaleData(ref_data, features = genes.leaveout)

ref_data <- as.SingleCellExperiment(ref_data)
query_data <- as.SingleCellExperiment(query_data)
rowData(ref_data)$feature_symbol <- rownames(ref_data)
rowData(query_data)$feature_symbol <- rownames(query_data)


acc <- c()
f1 <- c()
f1w <- c()
## scmap-cell
ref_data <- indexCell(ref_data)
scmapCell_results <- scmapCell(projection = query_data,list(ref = metadata(ref_data)$scmap_cell_index))
scmapCell_clusters <- scmapCell2Cluster(scmapCell_results,list(as.character(colData(ref_data)$celltype)))
result <- scmapCell_clusters$scmap_cluster_labs

y_true <- query_data$celltype
y_pred <- result

print(table(y_true, y_pred))
acc <- append(acc, Accuracy(y_true, y_pred))
f1 <- append(f1, F1_Score_macro(y_true, y_pred))
f1w <- append(f1w, F1_Score_macro_weighted(y_true, y_pred))

print(acc)
print(f1)
print(f1w)
