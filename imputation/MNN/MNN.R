# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.08.17               #
# ***********************************

library(Seurat)
library(ggplot2)
library(rjson)
library(batchelor)
library(Matrix)
library(BiocParallel)
library(BiocNeighbors)
library(igraph)
library(reshape2)
library(knitr)
library(scater)
library(scales)
library(irlba)
library(tibble)
library(dplyr)

# Moffit RNA
Moffit <- Read10X("/data/dataset/hypothalamic_merfish/GSE113576/")
Mo <- CreateSeuratObject(counts = Moffit, project = 'POR', min.cells = 10)
Mo <- NormalizeData(object = Mo, scale.factor = 1000000)

Moffit <- as.matrix(Mo@assays$RNA@counts)
Moffit_norm <- as.matrix(Mo@assays$RNA@data)
# Genes_count = rowSums(Moffit > 0)
# Moffit <- Moffit[Genes_count>=10,]

# MERFISH
MERFISH <- read.csv(file = "/data/dataset/hypothalamic_merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv", header = TRUE)
MERFISH_1 <- MERFISH[MERFISH['Animal_ID']==1,]
MERFISH_1 <- MERFISH_1[MERFISH_1['Cell_class']!='Ambiguous',]
MERFISH_meta <- MERFISH_1[,c(1:9)]
MERFISH_data <- MERFISH_1[,c(10:170)]
drops <- c('Blank_1','Blank_2','Blank_3','Blank_4','Blank_5','Fos')
MERFISH_data <- MERFISH_data[ , !(colnames(MERFISH_data) %in% drops)]
MERFISH_data <- t(MERFISH_data)

Gene_set <- intersect(rownames(MERFISH_data),rownames(Moffit))
Gene_groups <- fromJSON(file = "../gimvi/data/gene_groups.json")

# order scRNA-seq and SRT
Moffit = Moffit[Gene_set,]
MERFISH_data = MERFISH_data[Gene_set,]
cosineNorm_RNA <- cosineNorm(Moffit)
cosineNorm_MERFISH <- cosineNorm(MERFISH_data)

Imputed_genes <- matrix(0,nrow = length(Gene_set),ncol = ncol(MERFISH_data))
rownames(Imputed_genes) <- Gene_set
colnames(Imputed_genes) = colnames(MERFISH_data)

MNN_time <- vector(mode= "numeric")
Transfer_time <- vector(mode= "numeric")

for(i in 1:length(Gene_groups)) {
  print(i)
  print(Gene_groups[[i]])
  start_time <- Sys.time()

  # Mapping
  cosineNorm_MERFISH_leaveout = cosineNorm_MERFISH[-which(rownames(cosineNorm_MERFISH) %in% Gene_groups[[i]]),]
  cosineNorm_RNA_leaveout = cosineNorm_RNA[-which(rownames(cosineNorm_RNA) %in% Gene_groups[[i]]),]
  mbpca = multiBatchPCA(cosineNorm_MERFISH_leaveout, cosineNorm_RNA_leaveout, d=50)
  out = do.call(reducedMNN, mbpca)
  common_pca = out$corrected
  current_knns = queryKNN( common_pca[colnames(Moffit),], common_pca[colnames(MERFISH_data),], k = 25, get.index = TRUE, get.distance = FALSE)
  cells_mapped = apply(current_knns$index, 1, function(x) colnames(Moffit)[x])
  rownames(cells_mapped) = paste0("Fold", i, "_", c(1:25))
  end_time <- Sys.time()
  MNN_time <- c(MNN_time,as.numeric(difftime(end_time,start_time,units = 'secs')))

  # Imputation
  start_time <- Sys.time()
  for (j in 1:length(Gene_groups[[i]])) {
    RNA_gene_reference = Moffit_norm[rownames(Moffit_norm) == Gene_groups[[i]][[j]],]
    for (k in 1:ncol(cells_mapped)) {
      Imputed_genes[Gene_groups[[i]][[j]],k] = mean( RNA_gene_reference[cells_mapped[,k]] )
    }
  }
  end_time <- Sys.time()
  Transfer_time <- c(Transfer_time,as.numeric(difftime(end_time,start_time,units = 'secs')))
}

write.csv(Imputed_genes,file = 'Results/MNN_FiveFolds.csv')
write.csv(MNN_time,file = 'Results/MNN_time.csv',row.names = FALSE)
write.csv(Transfer_time,file = 'Results/Transfer_time.csv',row.names = FALSE)
