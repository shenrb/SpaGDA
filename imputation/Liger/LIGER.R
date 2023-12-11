# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.17               #
# ***********************************

setwd("/data/imputation/control_methods/Liger/")

library(liger)
library(Seurat)
library(ggplot2)
library(rjson)

# Moffit RNA
Moffit <- Read10X("/data/dataset/hypothalamic_merfish/GSE113576/")
Moffit <- as.matrix(Moffit)
Genes_count = rowSums(Moffit > 0)
Moffit <- Moffit[Genes_count>=10,]

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
Gene_groups <- fromJSON(file = "data/gene_groups.json")

#### New genes prediction
Ligerex <- createLiger(list(MERFISH = MERFISH_data, Moffit_RNA = Moffit))
Ligerex <- normalize(Ligerex)
Ligerex@var.genes <- Gene_set
Ligerex <- scaleNotCenter(Ligerex)

# suggestK(Ligerex) # K = 25
# suggestLambda(Ligerex, k = 25)

Ligerex <- optimizeALS(Ligerex,k = 25)
Ligerex <- quantileAlignSNF(Ligerex)

# Five Folds validation
Imputed_genes <- matrix(0,nrow = length(Gene_set),ncol = dim(MERFISH_data)[2])
rownames(Imputed_genes) <- Gene_set
colnames(Imputed_genes) <- colnames(MERFISH_data)
NMF_time <- vector(mode= "numeric")
knn_time <- vector(mode= "numeric")

for(i in 1:length(Gene_groups)) {
  print(i)
  print(Gene_groups[[i]])
  start_time <- Sys.time()
  Ligerex.leaveout <- createLiger(list(MERFISH = MERFISH_data[-which(rownames(MERFISH_data) %in% Gene_groups[[i]]),], Moffit_RNA = Moffit))
  Ligerex.leaveout <- normalize(Ligerex.leaveout)
  Ligerex.leaveout@var.genes <- setdiff(Gene_set,Gene_groups[[i]])
  Ligerex.leaveout <- scaleNotCenter(Ligerex.leaveout)
  Ligerex.leaveout <- optimizeALS(Ligerex.leaveout,k = 25)
  Ligerex.leaveout <- quantileAlignSNF(Ligerex.leaveout)
  end_time <- Sys.time()
  NMF_time <- c(NMF_time,as.numeric(difftime(end_time,start_time,units = 'secs')))

  start_time <- Sys.time()
  Imputation <- imputeKNN(Ligerex.leaveout, reference = 'Moffit_RNA', queries = list('MERFISH'), norm = TRUE, scale = FALSE)
  for (j in 1:length(Gene_groups[[i]])) {
    Imputed_genes[Gene_groups[[i]][[j]],] = as.vector(Imputation@norm.data$MERFISH[Gene_groups[[i]][[j]],])
  }
  end_time <- Sys.time()
  knn_time <- c(knn_time,as.numeric(difftime(end_time,start_time,units = 'secs')))
}
write.csv(Imputed_genes,file = 'Results/Liger_FiveFolds.csv')
write.csv(NMF_time,file = 'Results/Liger_NMF_time.csv',row.names = FALSE)
write.csv(knn_time,file = 'Results/Liger_knn_time.csv',row.names = FALSE)
