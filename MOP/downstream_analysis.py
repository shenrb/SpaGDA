#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   shen_rongbo@gzlab.ac.cn  #
# Date:    2023.03.28               #
# ***********************************

import os
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as io
import scanpy as sc
from anndata import AnnData
import json
import SpatialDE
from scipy.cluster.hierarchy import leaves_list

def horizontal(array, height=0):
    horizonx = array[:,0]
    horizony = height - array[:,1]
    return np.stack((horizonx, horizony),axis=1)

def verizontal(array, width=0):
    verizonx = width - array[:,0]
    verizony = array[:,1]
    return np.stack((verizonx, verizony),axis=1)

def rotate(array, angle):
    rotatex = math.cos(angle)*array[:,0] - math.sin(angle)*array[:,1]
    rotatey = math.cos(angle)*array[:,1] + math.sin(angle)*array[:,0]
    return np.stack((rotatex, rotatey),axis=1)

def alignment(array):
    minx, maxx, miny, maxy = np.min(array[:,0]), np.max(array[:,0]), np.min(array[:,1]), np.max(array[:,1])
    return array - np.array([minx, miny]), maxx-minx, maxy-miny

def draw_axs_spatial_genes(ax, expression_values, x_locs, y_locs, title, ms, ylabel):

    ax.set_aspect(1)
    vmin = np.percentile(expression_values, 1)
    vmax = np.percentile(expression_values, 99)

    tmp = sorted(list(zip(expression_values,x_locs,y_locs)))
    expression_values, x_locs, y_locs = list(zip(*tmp))

    sca = ax.scatter(x=x_locs, y=y_locs, s=ms, c=expression_values, vmin=vmin, vmax=vmax, cmap='jet', edgecolors='none')

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    if ylabel: ax.set_ylabel(ylabel, fontsize=40, labelpad=10)
    if title: ax.set_title(title.split('_')[-1], fontsize=40, pad=10)
    return sca, vmin, vmax

def draw_visual_genes_mop(adata, sub_save_dir, ext, genes):
    #### Visual SpaGDA imputation result in spatial

    slices = list(set(adata.obs.slice_id))
    slices.sort()
    print("All slices:", slices)
    #load the imputed gene values


    # Plot the gene expression values on top of positions
    fig, axs = plt.subplots(len(genes), 4, figsize=(27,7*len(genes)), gridspec_kw={'hspace':0, 'wspace':0, 'width_ratios':np.array([1,1.6,0.8,1])})
    for i,nslice in enumerate(slices):
        slice_adata = adata[adata.obs.slice_id==nslice, :]
        coords = np.array(slice_adata.obsm['spatial'])
        _coords, _, _ =  alignment(coords)
        if nslice == 'mouse1_slice112':
            _coords, _, _ = alignment(verizontal(horizontal(_coords)))
        elif nslice =='mouse1_slice122' or nslice =='mouse1_slice131':
            _coords, _, _ = alignment(horizontal(_coords))            
        x_locs, y_locs = _coords[:,0], _coords[:,1]
        for j,gene in enumerate(genes):
            if j == 0 and i == 0:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata[:,gene].X, x_locs, y_locs, nslice, 10, gene)
            elif j == 0 and i != 0:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata[:,gene].X, x_locs, y_locs, nslice, 10, None)
            elif j != 0 and i == 0:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata[:,gene].X, x_locs, y_locs, None, 10, gene)
            else:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata[:,gene].X, x_locs, y_locs, None, 10, None)

    cb_ax = fig.add_axes([0.4, 0.08, 0.2, 0.06/len(genes)])
    cbar = fig.colorbar(sca, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cbar.set_ticklabels(['low', 'medium', 'high'])#, fontsize=20)

    fig.savefig(os.path.join(sub_save_dir, '%s.pdf'%ext), bbox_inches='tight', dpi=100)
    plt.close()

def draw_visual_module(adata, sub_save_dir, ext, modules):
    #### Visual SpaGDA imputation result in spatial

    slices = list(set(adata.obs.slice_id))
    slices.sort()
    print("All slices:", slices)
    #load the imputed gene values


    # Plot the modules on top of positions
    fig, axs = plt.subplots(len(modules), 4, figsize=(27,7*len(modules)), gridspec_kw={'hspace':0, 'wspace':0, 'width_ratios':np.array([1,1.6,0.8,1])})
    for i,nslice in enumerate(slices):
        slice_adata = adata[adata.obs.slice_id==nslice, :]
        coords = np.array(slice_adata.obsm['spatial'])
        _coords, _, _ =  alignment(coords)
        if nslice == 'mouse1_slice112':
            _coords, _, _ = alignment(verizontal(horizontal(_coords)))
        elif nslice =='mouse1_slice122' or nslice =='mouse1_slice131':
            _coords, _, _ = alignment(horizontal(_coords))            
        x_locs, y_locs = _coords[:,0], _coords[:,1]
        for j,module in enumerate(modules):
            if j == 0 and i == 0:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata.obs[module], x_locs, y_locs, nslice, 10, module)
            elif j == 0 and i != 0:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata.obs[module], x_locs, y_locs, nslice, 10, None)
            elif j != 0 and i == 0:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata.obs[module], x_locs, y_locs, None, 10, module)
            else:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j,i], slice_adata.obs[module], x_locs, y_locs, None, 10, None)

    cb_ax = fig.add_axes([0.4, 0.08, 0.2, 0.06/len(modules)])
    cbar = fig.colorbar(sca, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cbar.set_ticklabels(['low', 'medium', 'high'])#, fontsize=20)

    fig.savefig(os.path.join(sub_save_dir, '%s.pdf'%ext), bbox_inches='tight', dpi=100)
    plt.close()


def draw_axs_spatial_layers(ax, cell_labels, x_locs, y_locs, nslice, ms, ylabel, legend):

    ax.set_aspect(1)
    cell_type_pal = {'L2/3':'limegreen', 'L4':'darkcyan', 'L5':'purple', 'L6':'peru', 'L6b':'slateblue', 'other':'lightgray'}

    cell_types = ['other', 'L2/3', 'L4', 'L5', 'L6', 'L6b']
    tmp = list(zip(cell_labels, x_locs, y_locs))
    for cell_t in cell_types:
        sub_tmp = [i for i in tmp if i[0]==cell_t]
        sub_cell_labels, sub_x_locs, sub_y_locs = list(zip(*sub_tmp))
        ax.scatter(x=sub_x_locs, y=sub_y_locs, s=ms, c=cell_type_pal[cell_t], edgecolors='none', label=cell_t)

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    if legend: ax.legend(bbox_to_anchor=(1,0.5),loc=6, fontsize=30, borderaxespad=0, markerscale=8, edgecolor='white')
    if ylabel: ax.set_ylabel(ylabel, fontsize=40, labelpad=10)
    if nslice: ax.set_title(nslice.split('_')[-1], fontsize=40, pad=10)


def draw_visual_genes_mop_and_layer_slice(adata, sub_save_dir, ext, genes):
    #### Visual SpaGDA imputation result in spatial

    slices = list(set(adata.obs.slice_id))
    slices.sort()
    print("All slices:", slices)

    source_cell_types = ['L2/3 IT','L4/5 IT','L5 IT','L6 IT','L5 ET','L5/6 NP','L6 CT','L6b','L6 IT Car3']
    target_cell_types = ['L2/3','L4','L5','L6','L5','L5','L6','L6b','L6']
    cell_types_in_layers = list(adata.obs['subclass'].replace(source_cell_types, target_cell_types))
    adata.obs['cell_labels'] = [i if i in set(target_cell_types) else 'other' for i in cell_types_in_layers]


    slice_adata = adata[adata.obs.slice_id=='mouse1_slice153', :]
    coords = np.array(slice_adata.obsm['spatial'])
    _coords, _, _ =  alignment(coords)          
    x_locs, y_locs = _coords[:,0], _coords[:,1]

    fig, ax = plt.subplots(figsize=(5,8))
    draw_axs_spatial_layers(ax, slice_adata.obs.cell_labels, x_locs, y_locs, 'Layers', 15, None, True)
    fig.savefig(os.path.join(sub_save_dir, 'Layers.pdf'), bbox_inches='tight', dpi=100)
    plt.close()

    # Plot the gene expression values on top of positions
    fig, axs = plt.subplots(1, 5, figsize=(25,8), gridspec_kw={'hspace':0, 'wspace':0})

    for j,gene in enumerate(genes):
        sca, vmin, vmax = draw_axs_spatial_genes(axs[j], slice_adata[:,gene].X, x_locs, y_locs, gene, 15, None)

    cb_ax = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(sca, cax=cb_ax, orientation='vertical') #horizontal
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cbar.set_ticklabels(['low', 'medium', 'high'])#, fontsize=30)

    fig.savefig(os.path.join(sub_save_dir, '%s.pdf'%ext), bbox_inches='tight', dpi=100)
    plt.close()


def draw_visual_genes_mop_and_layer(adata, sub_save_dir, ext, genes):
    #### Visual SpaGDA imputation result in spatial

    slices = list(set(adata.obs.slice_id))
    slices.sort()
    print("All slices:", slices)

    source_cell_types = ['L2/3 IT','L4/5 IT','L5 IT','L6 IT','L5 ET','L5/6 NP','L6 CT','L6b','L6 IT Car3']
    target_cell_types = ['L2/3','L4','L5','L6','L5','L5','L6','L6b','L6']
    cell_types_in_layers = list(adata.obs['subclass'].replace(source_cell_types, target_cell_types))
    adata.obs['cell_labels'] = [i if i in set(target_cell_types) else 'other' for i in cell_types_in_layers]


    # Plot the gene expression values on top of positions
    fig, axs = plt.subplots(len(genes)+1, 4, figsize=(23,6*(1+len(genes))), gridspec_kw={'hspace':0, 'wspace':0, 'width_ratios':np.array([1,1.6,0.8,1])})
    for i,nslice in enumerate(slices):
        slice_adata = adata[adata.obs.slice_id==nslice, :]
        coords = np.array(slice_adata.obsm['spatial'])
        _coords, _, _ =  alignment(coords)
        if nslice == 'mouse1_slice112':
            _coords, _, _ = alignment(verizontal(horizontal(_coords)))
        elif nslice =='mouse1_slice122' or nslice =='mouse1_slice131':
            _coords, _, _ = alignment(horizontal(_coords))            
        x_locs, y_locs = _coords[:,0], _coords[:,1]

        if i == 0:
            draw_axs_spatial_layers(axs[0,i], slice_adata.obs.cell_labels, x_locs, y_locs, nslice, 15, 'Layers', None)
        elif i == 3:
            draw_axs_spatial_layers(axs[0,i], slice_adata.obs.cell_labels, x_locs, y_locs, nslice, 15, None, True)
        else:
            draw_axs_spatial_layers(axs[0,i], slice_adata.obs.cell_labels, x_locs, y_locs, nslice, 15, None, None)

        for j,gene in enumerate(genes):
            if i == 0:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j+1,i], slice_adata[:,gene].X, x_locs, y_locs, None, 15, gene)
            else:
                sca, vmin, vmax = draw_axs_spatial_genes(axs[j+1,i], slice_adata[:,gene].X, x_locs, y_locs, None, 15, None)

    cb_ax = fig.add_axes([0.91, 0.3, 0.02, 0.2])
    cbar = fig.colorbar(sca, cax=cb_ax, orientation='vertical') #horizontal
    cbar.ax.tick_params(labelsize=30)
    cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cbar.set_ticklabels(['low', 'medium', 'high'])#, fontsize=30)

    fig.savefig(os.path.join(sub_save_dir, '%s.pdf'%ext), bbox_inches='tight', dpi=100)
    plt.close()


def select_top_variable_genes(data_mtx, top_k):
    """
    data_mtx: data matrix (cell by gene)
    top_k: number of highly variable genes to choose
    """
    var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind




# draw DEG genes dotplot for raw merfish st dataset
st_file = 'data/MERFISH.h5ad'
adata = sc.read(st_file)
markers = ['Slc17a7', 'Slc32a1', 'Slc30a3', 'Cux2', 'Otof', 'Rorb',
           'Rspo1', 'Sulf2', 'Fezf2', 'Osr1', 'Car3', 'Fam84b',
           'Syt6', 'Cplx3', 'Tshz2', 'Pvalb', 'Sst', 'Vip', 'Sncg',
           'Lamp5', 'Sox10', 'Pdgfra', 'Gfap', 'Aqp4', 'Igf2', 'Cd14',
           'Ctss', 'Mrc1', 'Flt1', 'Acta2', 'Kcnj8']

save_dir = 'degs'
if not os.path.exists(save_dir): os.makedirs(save_dir)
sc.settings.figdir = save_dir

SpaGDA_imputed_sweden = pd.read_csv('data/gan_DAGAN_FiveFolds_2022-10-14-16-57-16.csv', header=0, index_col=0, sep=',')
genes = SpaGDA_imputed_sweden.columns.to_list()

var = pd.DataFrame(index=genes)
sweden_imputed_adata = AnnData(SpaGDA_imputed_sweden.to_numpy(), obs=adata.obs, var=var, dtype='float32')
sweden_imputed_adata.obsm['spatial'] = adata.obsm['spatial']

adata = adata[~adata.obs.cell_types.isin(['other']),:]
adata.obs['subclass'] = adata.obs['cell_types']
sc.pl.dotplot(adata, markers, groupby='subclass', dendrogram=True, swap_axes=False, save='markers.pdf')


# draw marker genes with sweden database
genes_set1 = ['Nxph3', 'Rprm', 'Coro6', 'Krt12', 'Gm12371']
draw_visual_genes_mop_and_layer(sweden_imputed_adata, save_dir, 'sweden_main', genes_set1)
draw_visual_genes_mop_and_layer_slice(sweden_imputed_adata, save_dir, 'sweden_main_153', genes_set1)
genes_set2 = ['Bmp3', 'Sstr2', 'Scube1', 'Tnnc1', 'A830009L08Rik']
draw_visual_genes_mop_and_layer(sweden_imputed_adata, save_dir, 'sweden_extend', genes_set2)
genes_set3 = ['3110035E14Rik', 'Tbr1', 'Hs3st4', 'Nov', 'Hs3st2', 'Dkkl1', 'A830036E02Rik', 'Rasl10a']
draw_visual_genes_mop_and_layer(sweden_imputed_adata, save_dir, 'sweden_other', genes_set3)
# draw dotplot for gene set1
source_cell_types = ['L2/3 IT','L4/5 IT','L5 IT','L6 IT','L5 ET','L5/6 NP','L6 CT','L6b','L6 IT Car3']
target_cell_types = ['L2/3','L4','L5','L6','L5','L5','L6','L6b','L6']
sub_imputed_it_adata = sweden_imputed_adata[sweden_imputed_adata.obs.subclass.isin(source_cell_types),:]
sub_imputed_it_adata.obs['Layers']  = list(sub_imputed_it_adata.obs['subclass'].replace(source_cell_types, target_cell_types))
sc.pl.dotplot(sub_imputed_it_adata, genes_set1, groupby='Layers', swap_axes=False, save='IT_layers.pdf')
sc.pl.stacked_violin(sub_imputed_it_adata, genes_set1, groupby='Layers', swap_axes=False, save='IT_layers.pdf')#cmap='jet', 
sc.pl.matrixplot(sub_imputed_it_adata, genes_set1, groupby='Layers', swap_axes=False, save='IT_layers.pdf')#cmap='RdBu_r'
sc.pl.tracksplot(sub_imputed_it_adata, genes_set1, groupby='Layers', swap_axes=False, save='IT_layers.pdf')

sub_imputed_it_adata.write('Results/sweden_imputed.h5ad')


# load imputed merfish st dataset.
imputed_st_file = 'Results/imputed_merfish_m1s3.h5ad'
imputed_adata = sc.read(imputed_st_file)
imputed_adata.obs['subclass'] = list(imputed_adata.obs['cell_types'])
raw_imputed_adata = imputed_adata.copy()
imputed_adata = imputed_adata[~imputed_adata.obs.cell_types.isin(['other']),:]


check_cell_types = ['L2/3 IT','L4/5 IT','L5 IT','L6 IT','L5 ET','L5/6 NP','L6 CT','L6b','L6 IT Car3']
visual_imputed_adata = imputed_adata[imputed_adata.obs.subclass.isin(check_cell_types),:]

sc.tl.rank_genes_groups(imputed_adata, groupby='subclass', method='wilcoxon') 
sc.pl.rank_genes_groups_dotplot(imputed_adata, n_genes=2, save='imputed.pdf')
sc.tl.filter_rank_genes_groups(imputed_adata, min_in_group_fraction=0.5, max_out_group_fraction=0.3, min_fold_change=2)
sc.pl.rank_genes_groups_dotplot(imputed_adata, n_genes=4, key='rank_genes_groups_filtered', dendrogram=True, save='imputed_filter.pdf')

sc.tl.rank_genes_groups(visual_imputed_adata, groupby='subclass', method='wilcoxon') 
sc.pl.rank_genes_groups_dotplot(visual_imputed_adata, n_genes=2, save='imputed_visual.pdf')
sc.tl.filter_rank_genes_groups(visual_imputed_adata, min_in_group_fraction=0.5, max_out_group_fraction=0.3, min_fold_change=2)
sc.pl.rank_genes_groups_dotplot(visual_imputed_adata, n_genes=4, key='rank_genes_groups_filtered', dendrogram=True, save='imputed_visual_filter.pdf')


deg_markers = {'Lamp5':['Dlx6os1', 'Kit'],
                'Sncg': ['Gm29683', 'Crh'],
                'Vip': ['Igf1', 'Dlx1'],
                'Pvalb': ['Tmem132c', 'Tac1'],
                'Sst': ['Gm1604a', 'Cntnap3'],
                'L5 ET': ['Gm44593', 'Gm6260'],
                'L5/6 NP': ['Grp', 'Dkk2'],
                'L6 CT': ['Chrna5', 'Nxph3'],
                'L6b': ['AC124490.1', 'Tnmd'],
                'L4/5 IT': ['Gm20752', 'Col8a1'], 
                'L5 IT': ['Gm10635', 'Tnnc1'], 
                'L2/3 IT': ['Tmc1', 'Ptgs2'], 
                'L6 IT': ['6530403H02Rik'], 
                'L6 IT Car3' : ['C130073E24Rik', '4930555F03Rik'],
                'OPC': ['Vcan', 'Stk32a'], 
                'Oligo': ['AC110241.2', 'Hhip'],
                'Astro': ['Pdzph1', 'Gm35552'],
                'VLMC': ['Rbp1', 'Eya2'], 
                'Macrophage': ['Csf3r', 'Cysltr1'],
                'Endo': ['Ptprb', 'Pecam1'],
                'Peri': ['Ndufa4l2', 'Tbx3os1'],
                'PVM': ['Il1r1', 'Clec2d'],
                'SMC': ['Mgp', 'Uaca']
                }
sc.pl.dotplot(imputed_adata, deg_markers, groupby='subclass', dendrogram=True, swap_axes=False, save='deg.pdf')




# svg analysis
with open('data/sc_gene_folds.json') as f:
    gene_groups = json.load(f)
hvg_genes = []
for k,v in gene_groups.items():
    print('==> Fold:', k)
    print('==> Imputed genes:', v)
    hvg_genes  = hvg_genes + v

with open('data/sc_gene_variance_norm.json') as f:
    hvg_gene_variance_norms = json.load(f)




# draw demo genes with allen ish
main_genes = ['Ighm', 'Myl4', 'Hs3st2', 'Igsf21', 'Prkcg', 'Tnnc1', 'Gsg1l', 'Ddit4l', 'Car4', 'Rasgrf2', 'Syt17', 'Ogn']
demo_adata = raw_imputed_adata[raw_imputed_adata.obs.slice_id=='mouse1_slice153', :]

coords = np.array(demo_adata.obsm['spatial'])
_coords, _, _ =  alignment(coords)           
x_locs, y_locs = _coords[:,0], _coords[:,1]

fig, axs = plt.subplots(1, 12, figsize=(50,8), gridspec_kw={'hspace':0, 'wspace':0})

for j,gene in enumerate(main_genes):
    if j == 0:
        sca, vmin, vmax = draw_axs_spatial_genes(axs[j], demo_adata[:,gene].X, x_locs, y_locs, gene, 15, 'SpaGDA')
    else:
        sca, vmin, vmax = draw_axs_spatial_genes(axs[j], demo_adata[:,gene].X, x_locs, y_locs, gene, 15, None)

cb_ax = fig.add_axes([0.91, 0.3, 0.01, 0.4])
cbar = fig.colorbar(sca, cax=cb_ax)
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
cbar.set_ticklabels(['low', 'medium', 'high'])#, fontsize=15)
fig.savefig(os.path.join(save_dir, 'genes_compare_allen.pdf'), bbox_inches='tight', dpi=100)
plt.close()


fig, axs = plt.subplots(1, 6, figsize=(25,8), gridspec_kw={'hspace':0, 'wspace':0})
for j,gene in enumerate(main_genes[:6]):
    if j == 0:
        sca, vmin, vmax = draw_axs_spatial_genes(axs[j], demo_adata[:,gene].X, x_locs, y_locs, gene, 15, 'SpaGDA')
    else:
        sca, vmin, vmax = draw_axs_spatial_genes(axs[j], demo_adata[:,gene].X, x_locs, y_locs, gene, 15, None)

cb_ax = fig.add_axes([0.91, 0.3, 0.01, 0.4])
cbar = fig.colorbar(sca, cax=cb_ax)
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
cbar.set_ticklabels(['low', 'medium', 'high'])#, fontsize=15)
fig.savefig(os.path.join(save_dir, 'genes_compare_allen_sub1.pdf'), bbox_inches='tight', dpi=100)
plt.close()

fig, axs = plt.subplots(1, 6, figsize=(25,8), gridspec_kw={'hspace':0, 'wspace':0})
for j,gene in enumerate(main_genes[6:]):
    if j == 0:
        sca, vmin, vmax = draw_axs_spatial_genes(axs[j], demo_adata[:,gene].X, x_locs, y_locs, gene, 15, 'SpaGDA')
    else:
        sca, vmin, vmax = draw_axs_spatial_genes(axs[j], demo_adata[:,gene].X, x_locs, y_locs, gene, 15, None)

cb_ax = fig.add_axes([0.91, 0.3, 0.01, 0.4])
cbar = fig.colorbar(sca, cax=cb_ax)
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
cbar.set_ticklabels(['low', 'medium', 'high'])#, fontsize=15)
fig.savefig(os.path.join(save_dir, 'genes_compare_allen_sub2.pdf'), bbox_inches='tight', dpi=100)
plt.close()


def draw_umap(adata, save_dir, ext, annot, color_pal, pca=False, normal=True):
    plt.style.use('seaborn-white') 
    sc.settings.figdir = save_dir
    if normal:
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)

    if pca:
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata)
    else:
        sc.pp.neighbors(adata, use_rep='X')

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=annot, palette=color_pal, legend_fontsize='large', save="_%s.pdf"%ext)


# load output merfish st dataset.
output_st_file = 'Results/output_merfish_m1s3.h5ad'
output_adata = sc.read(output_st_file)
output_adata = output_adata[~output_adata.obs.cell_types.isin(['other']),:]
output_adata.obs['subclass'] = list(output_adata.obs['cell_types'])
output_adata.obs['Ground Truth'] = output_adata.obs['cell_types'].replace(['Micro'], ['Macrophage'])

color_dict = {'L2/3 IT': 'green', 'L5 IT': 'purple', 'L6 IT': 'pink', 'L5 ET': 'violet', 'L5/6 NP': 'yellowgreen', 'L6 CT': 'peru', 'L6b': 'slateblue', 
              'Lamp5': 'blue', 'Sncg': 'lavender', 'Vip': 'gold', 'Sst': 'red', 'Pvalb': 'indigo', 'Peri': 'linen', 'Endo': 'brown', 'SMC': 'thistle', 
              'VLMC': 'orange', 'Astro': 'deepskyblue', 'Macrophage': 'lightcyan', 'OPC': 'lightgreen', 'Oligo': 'limegreen', 'L4/5 IT': 'darkcyan', 
              'L6 IT Car3': 'skyblue', 'PVM': '#00b1c8', 'other': 'lightgray', 'unassigned': 'lightgray'}
print('Draw umap of imputed mop dataset with ground truth.')
draw_umap(output_adata.copy(), save_dir, ext='imputed_mop', annot='Ground Truth', color_pal=color_dict, pca=True, normal=False)

# SpatialDE for spatial variable gene analysis
sub_adata = output_adata[output_adata.obs.slice_id=='mouse1_slice153', :]

counts = pd.DataFrame(sub_adata.X, columns=sub_adata.var_names, index=sub_adata.obs_names)
coord = pd.DataFrame(sub_adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=sub_adata.obs_names)
results = SpatialDE.run(coord, counts)

fig, ax = plt.subplots(figsize=(10,8))
ax.set_yscale('log')
anchor_genes_df = results[~results['g'].isin(hvg_genes)]
imputed_genes_df = results[results['g'].isin(set(hvg_genes)-set(main_genes))]
demo_genes_df = results[results['g'].isin(main_genes)]
ax.scatter(x=imputed_genes_df['FSV'], y=imputed_genes_df['LLR'], s=15, c='grey', edgecolors='none', label='Imputed genes')
ax.scatter(x=anchor_genes_df['FSV'], y=anchor_genes_df['LLR'], s=15, c='limegreen', edgecolors='none', label='Measured genes')
ax.scatter(x=demo_genes_df['FSV'], y=demo_genes_df['LLR'], s=15, c='grey', edgecolors='none', label='Imputed genes')
ax.scatter(x=demo_genes_df['FSV'], y=demo_genes_df['LLR'], s=15, c='none', linewidths=0.5, edgecolors='red')

main_gene_set1 = ['Car4', 'Ogn', 'Ddit4l'] # up
main_gene_set2 = ['Prkcg', 'Syt17'] # right
main_gene_set3 = ['Igsf21', 'Gsg1l'] # bottom  'Hs3st2', 'Rasgrf2', 
for gene in main_gene_set1:
    ax.text(demo_genes_df.loc[demo_genes_df.g==gene, 'FSV'], demo_genes_df.loc[demo_genes_df.g==gene, 'LLR']*1.08, gene, 
            c='black', ha='center', va='bottom', fontstyle='italic', weight='bold', fontsize=10)
for gene in main_gene_set2:
    ax.text(demo_genes_df.loc[demo_genes_df.g==gene, 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g==gene, 'LLR'], gene, 
            c='black', ha='left', va='center', fontstyle='italic', weight='bold', fontsize=10)

ax.text(demo_genes_df.loc[demo_genes_df.g=='Myl4', 'FSV']*0.99, demo_genes_df.loc[demo_genes_df.g=='Myl4', 'LLR'], 'Myl4', 
        c='black', ha='right', va='center', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Tnnc1', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Tnnc1', 'LLR']*0.95, 'Tnnc1', 
        c='black', ha='left', va='top', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Igsf21', 'FSV']*0.99, demo_genes_df.loc[demo_genes_df.g=='Igsf21', 'LLR']*0.8, 'Igsf21', 
        c='black', ha='right', va='top', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Gsg1l', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Gsg1l', 'LLR']*0.70, 'Gsg1l', 
        c='black', ha='center', va='top', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Ighm', 'FSV']*0.99, demo_genes_df.loc[demo_genes_df.g=='Ighm', 'LLR']*1.04, 'Ighm', 
        c='black', ha='right', va='bottom', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Hs3st2', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Hs3st2', 'LLR']*1.6, 'Hs3st2', 
        c='black', ha='right', va='bottom', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Rasgrf2', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Rasgrf2', 'LLR']*1.01, 'Rasgrf2', 
        c='black', ha='left', va='bottom', fontstyle='italic', weight='bold', fontsize=10)

ax.set_xlabel('FSV', fontsize=25, labelpad=10) #Fraction spatial variance
ax.set_title('Imputed genes vs. Measured genes', fontsize=25, pad=10)
ax.set_ylabel('LLR', fontsize=25, labelpad=10)
ax.tick_params(labelsize=20, pad=5)
ax.spines[['top', 'right']].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
newLabels, newHandles = [], []
for handle, label in zip(handles, labels):
    if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
ax.legend(newHandles, newLabels, fontsize=20, markerscale=3, edgecolor='white')

fig.savefig(os.path.join(save_dir, 'FSV_LLR_compare.pdf'), bbox_inches='tight', dpi=100)
plt.close()


def draw_matrix_map(matrix, save_dir, name, label_x, label_y):
    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(5,5))
    ax = sns.heatmap(matrix, ax=ax, cbar=True, cbar_kws={'shrink':0.75}, annot=True, linewidths=0.5,
                     cmap='Reds', square=True, xticklabels=label_x, yticklabels=label_y)
    ax.tick_params(labelsize=20)
    ax.set_title('Spearman correlation', fontsize=25, pad=10)
    plt.yticks(rotation=0)
    plt.xticks(rotation=30, ha='right') 
    fig.savefig(os.path.join(save_dir, '%s.pdf'%name), bbox_inches='tight')
    plt.close()

# LLR correlation between slices
slices = list(set(output_adata.obs.slice_id))
slices.sort()

genes_sequence = []
llr_sequence = []
for i, nslice in enumerate(slices):
    sub_adata = output_adata[output_adata.obs.slice_id==nslice, :]

    counts = pd.DataFrame(sub_adata.X, columns=sub_adata.var_names, index=sub_adata.obs_names)
    coord = pd.DataFrame(sub_adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=sub_adata.obs_names)
    results = SpatialDE.run(coord, counts)

    sorted_results = results.sort_values('LLR', ascending=False)[['g','FSV','LLR','qval']]
    genes_sequence.append(sorted_results['g'].tolist())
    llr_sequence.append(sorted_results['LLR'].tolist())
    print('All genes in results: %d\nNon-sign genes number: %d'%(sorted_results.shape[0], sorted_results.query('qval < 0.05').shape[0]))

common_genes = set(genes_sequence[0]) & set(genes_sequence[1]) & set(genes_sequence[2]) & set(genes_sequence[3]) 

slice_names = [x.split('_')[-1] for x in slices]
llr_arr = np.zeros((4, len(common_genes)), dtype='float32')
for i in range(4):
    for j,g in enumerate(common_genes):
        idx = genes_sequence[i].index(g)
        llr_arr[i, j] = llr_sequence[i][idx]
llr_df = pd.DataFrame(llr_arr.T, columns=slice_names)
scc_df = llr_df.corr(method ='spearman')
draw_matrix_map(scc_df.to_numpy(), save_dir, 'LLR_corr', llr_df.columns, llr_df.columns)

from joypy import joyplot
joy_df = pd.DataFrame(columns=['LLR', 'Slice'])
joy_df['LLR'] = np.log10(llr_arr).ravel() # llr_arr.reshape(-1)
joy_df['Slice'] = sorted(slice_names*len(common_genes))
fig, axes = joyplot(joy_df, by='Slice', column='LLR', figsize=(6,3), linecolor='white', 
                    colormap=sns.color_palette('viridis', as_cmap=True), background='white')
#plt.xscale('log')
plt.xlim((0, 5))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(r'$log_{10}LLR$', fontsize=10, labelpad=5)
plt.title('LLR distributions', fontsize=20, pad=5)
fig.savefig(os.path.join(save_dir, 'LLR_joyplot.pdf'), bbox_inches='tight')
plt.close()

'''
#sign_results = results.query('qval < 0.05')
sign_results = results.sort_values('LLR', ascending=False).head(200)
#sign_results['l'].value_counts()
histology_results, patterns = SpatialDE.aeh.spatial_patterns(coord, counts, sign_results, C=10, l=10, verbosity=1)

for i in histology_results.sort_values('pattern').pattern.unique():
    print('Pattern {}'.format(i))
    print(', '.join(histology_results.query('pattern == @i').sort_values('membership')['g'].tolist()))
    print()

fig, axs = plt.subplots(2, 5, figsize=(40,15), gridspec_kw={'hspace':0, 'wspace':0.1})
x_locs, y_locs = coord['x_coord'], coord['y_coord']
for i in range(10):
    m,n = int(i/5), i%5

    axs[m,n].set_aspect(1)

    tmp = sorted(list(zip(patterns[i],x_locs,y_locs)))
    expression_values, x_locs, y_locs = list(zip(*tmp))

    sca = axs[m,n].scatter(x_locs, y_locs, s=10, c=expression_values, cmap='jet', edgecolors='none');
    axs[m,n].set_title('Pattern {} - {} genes'.format(i, histology_results.query('pattern == @i').shape[0] ))
    vmin = np.percentile(expression_values, 1)
    vmax = np.percentile(expression_values, 99)
    for sp in axs[m,n].spines.values():
        sp.set_visible(False)
    axs[m,n].set_xticks([])
    axs[m,n].set_yticks([])

cb_ax = fig.add_axes([0.91, 0.3, 0.01, 0.4])
cbar = fig.colorbar(sca, cax=cb_ax)
cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
cbar.set_ticklabels(['low', 'medium', 'high'], fontsize=15)
fig.savefig(os.path.join(save_dir, 'spatial_patterns.pdf'), bbox_inches='tight', dpi=100)
plt.close()
'''
'''
main_gene_set1 = ['Car4', 'Ogn', 'Ddit4l'] # up
main_gene_set2 = ['Prkcg', 'Syt17'] # right
main_gene_set3 = ['Igsf21', 'Gsg1l'] # bottom  'Hs3st2', 'Rasgrf2', 
for gene in main_gene_set1:
    ax.text(demo_genes_df.loc[demo_genes_df.g==gene, 'FSV'], demo_genes_df.loc[demo_genes_df.g==gene, 'LLR']*1.08, gene, 
            c='black', ha='center', va='bottom', fontstyle='italic', weight='bold', fontsize=10)
for gene in main_gene_set2:
    ax.text(demo_genes_df.loc[demo_genes_df.g==gene, 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g==gene, 'LLR'], gene, 
            c='black', ha='left', va='center', fontstyle='italic', weight='bold', fontsize=10)

ax.text(demo_genes_df.loc[demo_genes_df.g=='Myl4', 'FSV']*0.99, demo_genes_df.loc[demo_genes_df.g=='Myl4', 'LLR'], 'Myl4', 
        c='black', ha='right', va='center', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Tnnc1', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Tnnc1', 'LLR']*0.95, 'Tnnc1', 
        c='black', ha='left', va='top', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Igsf21', 'FSV']*0.99, demo_genes_df.loc[demo_genes_df.g=='Igsf21', 'LLR']*0.8, 'Igsf21', 
        c='black', ha='right', va='top', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Gsg1l', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Gsg1l', 'LLR']*0.70, 'Gsg1l', 
        c='black', ha='center', va='top', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Ighm', 'FSV']*0.99, demo_genes_df.loc[demo_genes_df.g=='Ighm', 'LLR']*1.04, 'Ighm', 
        c='black', ha='right', va='bottom', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Hs3st2', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Hs3st2', 'LLR']*1.6, 'Hs3st2', 
        c='black', ha='right', va='bottom', fontstyle='italic', weight='bold', fontsize=10)
ax.text(demo_genes_df.loc[demo_genes_df.g=='Rasgrf2', 'FSV']*1.02, demo_genes_df.loc[demo_genes_df.g=='Rasgrf2', 'LLR']*1.01, 'Rasgrf2', 
        c='black', ha='left', va='bottom', fontstyle='italic', weight='bold', fontsize=10)
'''


