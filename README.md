# Feature
SpaGDA is a unified deep learning-based graph domain adaptation for gene expression enhancement and cell type identification in large-scale spatial resolved transcriptomics. This repository includes three python sub-directories for gene expression enhancement, cell type identification and application, respectively. Developed by the author R. Shen (shen_rongbo@gzlab.ac.cn). 

![figure1](https://user-images.githubusercontent.com/8838722/221451761-1d73e37e-156f-43ee-9d39-7a5306b23540.png)

# Pre-requirements
All analyses presented in the paper were performed in a workstation with 40 Gb RAM memory, 10 cores of 2.5 GHz Intel Xeon Platinum 8255C CPU, and a Nvidia Tesla V100 GPU with 32 Gb memory. And the following python (v3.9) packages support for DAGCN are required: numpy==1.22.4, pandas==1.4.3, scipy==1.9.0, matplotlib==3.5.3, seaborn==0.11.2, scikit-learn==1.1.2, torch==1.21.1, scanpy==1.9.1, anndata==0.8.0.

# Data availability
The public datasets are freely available as follow. 

Mouse brain - primary motor cortex (MOp): doi:10.35077/g.21 (https://doi.brainimagelibrary.org/doi/10.35077/g.21).

Mouse brain - hypothalamic preoptic region (HPR): doi:10.5061/dryad.8t8s248 (https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248). 

Mouse spermatogenesis: (https://www.dropbox.com/s/ygzpj0d0oh67br0/Testis_Slideseq_Data.zip?dl=0). 

The mouse primary visual cortex (VISp) dataset measured by STARmap is available at (https://www.starmapresources.com/data), that is also available at Zenodo (https://doi.org/10.5281/zenodo.3967291). 

The snRNA-seq 10x v3 B of BICCN MOP dataset (RRID: SCR_015820) can be accessed via the NeMO archive (RRID: SCR_002001) at accession: (https://assets.nemoarchive.org/dat-ch1nqb7). 

scRNA-seq of preoptic region of mouse hypothalamic: GSE113576 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113576). 

scRNA-seq of mouse testis: GSE112393 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112393). 

The AllenVISp dataset (access code: GSE115746) is available at (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746), that is also available at Zenodo (https://doi.org/10.5281/zenodo.3967291).

# Usage
Train.py is the main program. 
```python
python3 Train.py
```
Option.py includes user-configurable parameters. For the experiments in this work, the configurations of involved paramenters are listed in the table.
![figure_table](https://user-images.githubusercontent.com/8838722/221455638-f0d9582f-648e-460b-a030-de0e7b341383.png)

# Acknowledgements
This work was supported by self-supporting program of Guangzhou Laboratory (Grant No. ZL-SRPG2200702).
