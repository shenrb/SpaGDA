# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com #
# Date:    2022.05.13               #
# ***********************************

setwd("/data/imputation/control_methods/Liger/")

# MERFISH
MERFISH <- read.csv(file = "/data/dataset/hypothalamic_merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv", header = TRUE)
MERFISH_1 <- MERFISH[MERFISH['Animal_ID']==1,]
MERFISH_1 <- MERFISH_1[MERFISH_1['Cell_class']!='Ambiguous',]
MERFISH_meta <- MERFISH_1[,c(1:9)]
MERFISH_data <- MERFISH_1[,c(10:170)]
drops <- c('Blank_1','Blank_2','Blank_3','Blank_4','Blank_5','Fos')
MERFISH_data <- MERFISH_data[ , !(colnames(MERFISH_data) %in% drops)]
MERFISH_data <- t(MERFISH_data)

write.csv(MERFISH_data,file = 'Data/MERFISH.csv')
