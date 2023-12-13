library(ggplot2)
library(tidyverse)
library(scibet)
library(viridis)
library(ggsci)
# library(dplyr)
# library(magrittr)
library(Matrix)
library(Seurat)
library(SeuratDisk)
library(patchwork)
library(MLmetrics)

data <- LoadH5Seurat("/aaa/jianhuayao/project2/data/Zheng68k/Zheng68k_full.h5seurat")
data <- NormalizeData(data)

acc <- c()
f1 <- c()
f1w <- c()
SEED <- 1
for (rep in 1:5) {
    set.seed(SEED + rep)
    data$k.assign <- sample(x = 1:5, size = ncol(data), replace = TRUE)
    ## Set 1 fold as query and other folds as reference
    for (i in 1:5) {
        ref_data <- subset(data,k.assign != i)
        query_data <- subset(data, k.assign == i)
        ref_matrix <- as.tbl(as.data.frame(t(GetAssayData(object = ref_data, slot = "data"))))
        query_matrix <- as.tbl(as.data.frame(t(GetAssayData(object = query_data, slot = "data"))))
        ref_matrix$label <- as.character(ref_data$celltype)
        query_matrix$label <- as.character(query_data$celltype)
        result <- SciBet(ref_matrix, query_matrix[,-ncol(query_matrix)])

        y_true <- as.character(query_data$celltype)
        y_pred <- as.character(result)
        print(table(y_true, y_pred))
        acc <- append(acc, Accuracy(y_true, y_pred))
        f1 <- append(f1, F1_Score_macro(y_true, y_pred))
        f1w <- append(f1w, F1_Score_macro_weighted(y_true, y_pred))
    }
    print(acc)
    print(f1)
    print(f1w)
}
print(acc)
print(f1)
print(f1w)
