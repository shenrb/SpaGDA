#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen@tencent.com   #
# Date:    2022.06.16               #
# **********************************#

import argparse
import os
import torch
import Utils
import time

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='mop', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='4', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # Model Generator parameters
        parser.add_argument('--model', type=str, default='gan', help='chooses which model to use. [cycle_gan | gan]')
        parser.add_argument('--input_dim', type=int, default=352, help='Dim of input, using HVG')
        parser.add_argument('--feat_hidden1', type=int, default=160, help='Dim of G-encoder hidden layer 1.')
        parser.add_argument('--feat_hidden2', type=int, default=80, help='Dim of G-encoder hidden layer 2.')
        parser.add_argument('--gcn_hidden1', type=int, default=40, help='Dim of G-VGAE hidden layer 1.')
        parser.add_argument('--gcn_hidden2', type=int, default=20, help='Dim of G-VGAE hidden layer 2.')
        parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
        parser.add_argument('--distance_type', type=str, default='euclidean', help='graph distance type: [euclidean | cosine | correlation].')

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--gene_groups', type=str,default='data/gene_folds.json', help='if specified, print more debugging information')


        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        _ = Utils.mk_dir(expr_dir)
        file_name = os.path.join(expr_dir, '%s_opt_%s.txt'%(opt.model,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
        opt.log_file = file_name
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    """This class includes training options.
       It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # Training data parameters
        parser.add_argument('--sc_data', type=str, default='data/snRNA.h5ad', help='source scRNA-seq dataset, h5ad format.')
        parser.add_argument('--st_data', type=str, default='data/MERFISH.h5ad', help='target spatial transcriptomic dataset, h5ad format.')

        # Discriminator parameters
        parser.add_argument('--dis_model', type=str, default='mlp', help='[mlp | conv]')
        parser.add_argument('--ndf', type=int, default=16, help='# of discriminator (D, netD) filters in the first conv layer.')
        parser.add_argument('--ndf_di', type=int, default=16, help='# of discriminator (D, netD) filters in the first conv layer.')
        parser.add_argument('--n_layers_D', type=int, default=1, help='netD n_layers')
        parser.add_argument('--n_layers_D_di', type=int, default=1, help='netD n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')

        # Training parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs with the initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [step | plateau | cosine]')
        parser.add_argument('--gamma', type=float, default=0.5, help='step lr decay gamma')
        parser.add_argument('--lr_decay_epochs', type=int, default=10, help='multiply by a gamma every lr_decay_epochs epochs')
        parser.add_argument('--cells_per_batch', type=int, default=400, help='random sampling #number of cells per epoch')
        parser.add_argument('--sc_neighbors', type=int, default=8, help='K nearest neighbors to create sc_graph.')
        parser.add_argument('--st_neighbors', type=int, default=15, help='K nearest neighbors to create st_graph.')
        parser.add_argument('--dis_sigma', type=int, default=40, help='dis sigma of nearest neighbors to weight st_graph.')

        ### Loss parameters
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--lambda_F', type=float, default=20, help='Weight of Autoencoder loss.')
        parser.add_argument('--lambda_G', type=float, default=0, help='Weight of VGAE loss.')
        parser.add_argument('--lambda_A', type=float, default=1, help='Weight for GAN loss.')
        parser.add_argument('--rec_loss', type=str, default='l2', help='rec loss type. [l2| l1 ].')

        ### Model saving and loading parameters

        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=40, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', type=bool, default=False, help='continue training: load the latest model')
        parser.add_argument('--print_freq', type=int, default=200, help='batch frequency of printing the loss')

        self.isTrain = True
        return parser


class TestOptions(BaseOptions):
    """This class includes test options.
       It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # Test data parameters
        parser.add_argument('--st_data', type=str, default='.h5ad', help='target spatial transcriptomic dataset, h5ad format.')

        # Test parameters
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        # Result saving parameters
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        self.isTrain = False
        return parser
