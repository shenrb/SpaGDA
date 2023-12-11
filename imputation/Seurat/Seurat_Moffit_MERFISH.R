# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.08.17               #
# ***********************************

setwd("/data/imputation/control_methods/Seurat/")
library(Seurat)
library(Matrix)

Moffit <- Read10X("/data/dataset/hypothalamic_merfish/GSE113576/")

Mo <- CreateSeuratObject(counts = Moffit, project = 'POR', min.cells = 10)
Mo <- NormalizeData(object = Mo, scale.factor = 1000000)
Mo <- FindVariableFeatures(object = Mo, nfeatures = 2000)
Mo <- ScaleData(object = Mo)
Mo <- RunPCA(object = Mo, npcs = 50, verbose = FALSE)
Mo <- RunUMAP(object = Mo, dims = 1:50, nneighbors = 5)
saveRDS(object = Mo, file = paste0("data/","Moffit_RNA.rds"))


MERFISH <- read.csv(file = "/data/dataset/hypothalamic_merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv", header = TRUE)
MERFISH_1 <- MERFISH[MERFISH['Animal_ID']==1,]
MERFISH_1 <- MERFISH_1[MERFISH_1['Cell_class']!='Ambiguous',]
MERFISH_meta <- MERFISH_1[,c(1:9)]
MERFISH_data <- MERFISH_1[,c(10:170)]
drops <- c('Blank_1','Blank_2','Blank_3','Blank_4','Blank_5','Fos')
MERFISH_data <- MERFISH_data[ , !(colnames(MERFISH_data) %in% drops)]
MERFISH_data <- t(MERFISH_data)

MERFISH_seurat <- CreateSeuratObject(counts = MERFISH_data, project = 'MERFISH', assay = 'RNA', meta.data = MERFISH_meta ,min.cells = -1, min.features = -1)
total.counts = colSums(x = as.matrix(MERFISH_seurat@assays$RNA@counts))
MERFISH_seurat <- NormalizeData(MERFISH_seurat, scale.factor = median(x = total.counts))
MERFISH_seurat <- ScaleData(MERFISH_seurat)
saveRDS(object = MERFISH_seurat, file = 'data/MERFISH.rds')
