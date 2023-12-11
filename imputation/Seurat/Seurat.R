# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.08.17               #
# ***********************************

library(Seurat)
library(ggplot2)
library(rjson)

MERFISH <- readRDS("data/MERFISH.rds")
Moffit <- readRDS("data/Moffit_RNA.rds")
Gene_groups <- fromJSON(file = "data/gene_groups.json")

genes.leaveout <- intersect(rownames(MERFISH),rownames(Moffit))
Imputed_genes <- matrix(0,nrow = length(genes.leaveout),ncol = dim(MERFISH@assays$RNA)[2])
rownames(Imputed_genes) <- genes.leaveout

anchor_time <- vector(mode= "numeric")
Transfer_time <- vector(mode= "numeric")

run_imputation <- function(ref.obj, query.obj, feature.remove) {
  #message(paste0('removing ', feature.remove))
  print("removing features... ")
  print(feature.remove)
  features <- setdiff(rownames(query.obj), feature.remove)
  DefaultAssay(ref.obj) <- 'RNA'
  DefaultAssay(query.obj) <- 'RNA'
  
  start_time <- Sys.time()
  anchors <- FindTransferAnchors(
    reference = ref.obj,
    query = query.obj,
    features = features,
    dims = 1:30,
    reduction = 'cca'
  )
  end_time <- Sys.time()
  print("anchor_time")
  print(difftime(end_time,start_time,units = 'secs'))
  anchor_time <- c(anchor_time,as.numeric(difftime(end_time,start_time,units = 'secs')))
  
  refdata <- GetAssayData(
    object = ref.obj,
    assay = 'RNA',
    slot = 'data'
  )
  
  start_time <- Sys.time()
  imputation <- TransferData(
    anchorset = anchors,
    refdata = refdata,
    weight.reduction = 'pca'
  )
  query.obj[['seq']] <- imputation
  end_time <- Sys.time()
  print("Transfer_time")
  print(difftime(end_time,start_time,units = 'secs'))
  Transfer_time <- c(Transfer_time,as.numeric(difftime(end_time,start_time,units = 'secs')))
  return(query.obj)
}

for(i in 1:length(Gene_groups)) {
  imputed.ss2 <- run_imputation(ref.obj = Moffit, query.obj = MERFISH, feature.remove = Gene_groups[[i]])
  MERFISH[['ss2']] <- imputed.ss2[, colnames(MERFISH)][['seq']]
  Imputed_genes[Gene_groups[[i]],] = as.vector(MERFISH@assays$ss2[Gene_groups[[i]],])
}

write.csv(Imputed_genes,file = 'Results/Seurat_FiveFolds.csv')
#write.csv(anchor_time,file = 'Results/Seurat_anchor_time.csv',row.names = FALSE)
#write.csv(Transfer_time,file = 'Results/Seurat_transfer_time.csv',row.names = FALSE)
