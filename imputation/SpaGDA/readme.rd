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

# Training data parameters
    sc_data: source scRNA-seq dataset, h5ad format
    st_data: target spatial transcriptomic dataset, h5ad format

# Discriminator parameters
    dis_model: [mlp | conv]
    ndf: # of discriminator (D, netD) filters in the first conv layer
    ndf_di: # of discriminator (D, netD) filters in the first conv layer
    n_layers_D: netD n_layers
    n_layers_D_di: netD n_layers
    norm: instance normalization or batch normalization [instance | batch | none]

# Training parameters
    phase: train, val, test, etc
    n_epochs: number of epochs with the initial learning rate
    beta1: momentum term of adam
    lr: initial learning rate for adam
    lr_policy: learning rate policy. [step | plateau | cosine]
    gamma: step lr decay gamma
    lr_decay_epochs: multiply by a gamma every lr_decay_epochs epochs
    cells_per_batch: random sampling #number of cells per epoch
    sc_neighbors: K nearest neighbors to create sc_graph
    st_neighbors: K nearest neighbors to create st_graph
    dis_sigma: dis sigma of nearest neighbors to weight st_graph

# Loss parameters
    gan_mode: the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper
    lambda_F: Weight of Autoencoder loss
    lambda_G: Weight of VGAE loss
    lambda_A: Weight for GAN loss
    rec_loss: rec loss type. [l2| l1 ]

