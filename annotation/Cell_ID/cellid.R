# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.17               #
# ***********************************


library(tidyverse)
library(Matrix)
library(Seurat)
library(SeuratDisk)
library(patchwork)
library(MLmetrics)
library(CellID)

query_data <- readRDS("MERFISH.rds")
ref_data <- readRDS("Moffit_RNA.rds")
Gene_groups <- fromJSON(file = "gene_groups.json")
genes.leaveout <- intersect(rownames(query_data),rownames(ref_data))

query_data <- ScaleData(query_data, features = genes.leaveout)
ref_data <- ScaleData(ref_data, features = genes.leaveout)


ref_data <- RunMCA(ref_data)
ref_cell_gs <- GetCellGeneSet(ref_data, dim=1:50, n.features=200)
ref_group_gs <- GetGroupGeneSet(ref_data, dim=1:50, n.features=200, group.by="celltype")
query_data <- FindVariableFeatures(query_data)
query_data <- RunMCA(query_data)

acc <- c()
f1 <- c()
f1w <- c()

## cell-to-cell match
HGT_ref_cell_gs <- RunCellHGT(query_data, pathways = ref_cell_gs, dims = 1:50)
ref_cell_gs_match <- rownames(HGT_ref_cell_gs)[apply(HGT_ref_cell_gs, 2, which.max)]
ref_cell_gs_prediction <- ref_data$celltype[ref_cell_gs_match]
ref_cell_gs_prediction_signif <- ifelse(apply(HGT_ref_cell_gs, 2, max)>2, yes = as.character(ref_cell_gs_prediction), "unassigned")
query_data$cell_gs_prediction <- ref_cell_gs_prediction_signif
result <- query_data$cell_gs_prediction

y_true <- as.character(query_data$celltype)
y_pred <- as.character(result)
print(table(y_true, y_pred))
acc <- append(acc, MLmetrics::Accuracy(y_true, y_pred))
f1 <- append(f1, MLmetrics::F1_Score_macro(y_true, y_pred))
f1w <- append(f1w, MLmetrics::F1_Score_macro_weighted(y_true, y_pred))

