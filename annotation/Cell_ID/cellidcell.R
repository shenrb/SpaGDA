library(tidyverse)
library(Matrix)
library(Seurat)
library(SeuratDisk)
library(patchwork)
library(MLmetrics)
library(CellID)

data <- LoadH5Seurat("/aaa/jianhuayao/project2/data/Zheng68k/Zheng68k_full.h5seurat")
data <- NormalizeData(data)

acc <- c()
f1 <- c()
f1w <- c()
SEED <- 1
for (rep in 1:5) {
    set.seed(SEED + rep)
    data$k.assign <- sample(x = 1:5, size = ncol(data), replace = TRUE)
    for (i in 1:5) {
        ref_data <- subset(data,k.assign != i)
        query_data <- subset(data, k.assign == i)
        ref_data <- ScaleData(ref_data, features = rownames(ref_data))
        ref_data <- RunMCA(ref_data)
        ref_cell_gs <- GetCellGeneSet(ref_data, dim=1:50, n.features=200)
        ref_group_gs <- GetGroupGeneSet(ref_data, dim=1:50, n.features=200, group.by="celltype")
        query_data <- FindVariableFeatures(query_data)
        query_data <- ScaleData(query_data)
        query_data <- RunMCA(query_data)

        ## cell-to-cell match
        HGT_ref_cell_gs <- RunCellHGT(query_data, pathways = ref_cell_gs, dims = 1:50)
        ref_cell_gs_match <- rownames(HGT_ref_cell_gs)[apply(HGT_ref_cell_gs, 2, which.max)]
        ref_cell_gs_prediction <- ref_data$celltype[ref_cell_gs_match]
        ref_cell_gs_prediction_signif <- ifelse(apply(HGT_ref_cell_gs, 2, max)>2, yes = as.character(ref_cell_gs_prediction), "unassigned")
        query_data$cell_gs_prediction <- ref_cell_gs_prediction_signif
        result <- query_data$cell_gs_prediction

        # ## group-to-cell match
        # HGT_ref_group_gs <- RunCellHGT(query_data, pathways = ref_group_gs, dims = 1:50)
        # ref_group_gs_prediction <- rownames(HGT_ref_group_gs)[apply(HGT_ref_group_gs, 2, which.max)]
        # ref_group_gs_prediction_signif <- ifelse(apply(HGT_ref_group_gs, 2, max)>2, yes = as.character(ref_group_gs_prediction), "unassigned")
        # query_data$group_gs_prediction <- ref_group_gs_prediction_signif
        # result <- query_data$group_gs_prediction

        # y_true <- vector()
        # y_pred <- vector()
        # for (j in 1:length(result)) {
        #     if (result[j] != "unassigned" && result[j] != "Unassigned" && result[j] != "unknown" && result[j] != "Unknown") {
        #         y_true <- append(y_true, as.character(query_data$celltype[j]))
        #         y_pred <- append(y_pred, result[j])
        #     }
        # }

        y_true <- as.character(query_data$celltype)
        y_pred <- as.character(result)
        print(table(y_true, y_pred))
        acc <- append(acc, MLmetrics::Accuracy(y_true, y_pred))
        f1 <- append(f1, MLmetrics::F1_Score_macro(y_true, y_pred))
        f1w <- append(f1w, MLmetrics::F1_Score_macro_weighted(y_true, y_pred))
    }
    print(acc)
    print(f1)
    print(f1w)
}
print(acc)
print(f1)
print(f1w)
# print(unl)