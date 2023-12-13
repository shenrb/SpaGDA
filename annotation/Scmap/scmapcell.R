library(dplyr)
library(magrittr)
library(Seurat)
library(SeuratDisk)
library(patchwork)
library(MLmetrics)
library(SingleCellExperiment)
library(scmap)

data <- LoadH5Seurat("/aaa/jianhuayao/project2/data/Zheng68k/Zheng68k_full.h5seurat")
data <- NormalizeData(data)
# data <- NormalizeData(object = data, normalization.method = "RC", scale.factor = 10000)
data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = 500)
slct <- VariableFeatures(object = data)

acc <- c()
f1 <- c()
f1w <- c()
SEED <- 1
for (rep in 1:5) {
    set.seed(SEED + rep)
    data$k.assign <- sample(x = 1:5, size = ncol(data), replace = TRUE)
    # unl <- 0
    for (i in 1:5) {
        ref_data <- subset(data,k.assign != i)
        query_data <- subset(data, k.assign == i)
        ref_data <- as.SingleCellExperiment(ref_data)
        query_data <- as.SingleCellExperiment(query_data)
        # logcounts(ref_data) <- log2(logcounts(ref_data)+1)
        # logcounts(query_data) <- log2(logcounts(query_data)+1)
        rowData(ref_data)$feature_symbol <- rownames(ref_data)
        rowData(query_data)$feature_symbol <- rownames(query_data)
        ref_data <- setFeatures(ref_data, slct)

        ## scmap-cell
        ref_data <- indexCell(ref_data)
        scmapCell_results <- scmapCell(projection = query_data,list(ref = metadata(ref_data)$scmap_cell_index))
        scmapCell_clusters <- scmapCell2Cluster(scmapCell_results,list(as.character(colData(ref_data)$celltype)))
        result <- scmapCell_clusters$scmap_cluster_labs

        # ## scmap-cluster
        # ref_data <- indexCluster(ref_data, cluster_col="celltype")
        # scmapCluster_results <- scmapCluster(projection = query_data,index_list = list(metadata(ref_data)$scmap_cluster_index))
        # result <- scmapCluster_results$scmap_cluster_labs

        # # y_true <- vector()
        # # y_pred <- vector()
        # # for (j in 1:length(result)) {
        # #     if (result[j] != "unassigned" && result[j] != "Unassigned") {
        # #         y_true <- append(y_true, as.character(query_data$celltype[j]))
        # #         y_pred <- append(y_pred, result[j])
        # #     }
        # # }
        y_true <- query_data$celltype
        y_pred <- result

        print(table(y_true, y_pred))
        acc <- append(acc, Accuracy(y_true, y_pred))
        f1 <- append(f1, F1_Score_macro(y_true, y_pred))
        f1w <- append(f1w, F1_Score_macro_weighted(y_true, y_pred))
        # unl <- append(unl, (1 - length(y_true) / length(query_data$celltype)))
    }
    print(acc)
    print(f1)
    print(f1w)
}
print(acc)
print(f1)
print(f1w)
# print(unl)

