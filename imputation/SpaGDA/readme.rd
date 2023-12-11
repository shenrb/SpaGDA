The project of SpaGDA for gene expression enhancement.


# basic parameters
    name: name of the experiment. It decides where to store samples and models
    gpu_ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
    checkpoints_dir: models are saved here

# Model Generator parameters
    model: chooses which model to use. [cycle_gan | gan]
    input_dim: Dim of input, using HVG
    feat_hidden1: Dim of G-encoder hidden layer 1
    feat_hidden2: Dim of G-encoder hidden layer 2
    gcn_hidden1: Dim of G-VGAE hidden layer 1
    gcn_hidden2: Dim of G-VGAE hidden layer 2
    p_drop: Dropout rate
    distance_type: graph distance type [euclidean | cosine | correlation]

# additional parameters
    verbose: if specified, print more debugging information
    load_epoch: which epoch to load? set to latest to use latest cached model
    gene_groups: if specified, print more debugging information
