#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***********************************
# Version: 0.1.1                    #
# Author:  rongboshen               #
# Email:   rongboshen2019@gmail.com # 
# Date:    2022.07.06               #
# ***********************************

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from VGAE import VGAEModel
from Discriminator import NLayerDiscriminator, NLayerMLP
from MLP import Basic_MLP
from Loss import GANLoss, GCNLoss, FocalLoss
from Utils import get_scheduler, get_norm_layer
import itertools
from collections import OrderedDict

class GANModel(nn.Module):
    """
    This class implements the GAN model.
    """

    def __init__(self, opt):
        """Initialize the GAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(GANModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        torch.backends.cudnn.benchmark = True

        self.loss_names = ['GAN', 'REC', 'DIS', 'CLS']
        self.optimizers = []
        self.metric = 0  # used for learning rate policy 'plateau'

        # define Generators
        self.netG = VGAEModel(opt).to(self.device)
        self.netC = Basic_MLP(opt).to(self.device)

        if self.isTrain:  # define discriminators
            if opt.dis_model == 'conv':
                self.netD = NLayerDiscriminator(ndf=opt.ndf, n_layers=opt.n_layers_D, kw=4, stride=2, norm_layer=get_norm_layer(opt.norm)).to(self.device)
            elif opt.dis_model == 'mlp':
                self.netD = NLayerMLP(input_dim=opt.feat_hidden2+opt.gcn_hidden2).to(self.device) 
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode, device=self.device)  # define GAN loss.
            self.criterionREC_F = torch.nn.MSELoss()                       # torch.nn.L1Loss()
            self.criterionREC_G = GCNLoss(device=self.device)              # (preds, labels, mu, logvar, n_nodes, norm)
            self.criterionCLS = FocalLoss()                                # define CLS loss.

            # initialize optimizers; schedulers will be automatically created.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) #RMSprop
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_C)

            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

            self.model_names = ['G', 'D', 'C']
        else: 
            self.model_names = ['G', 'C']

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def backward_D(self, x_s, graph_dict_s, x_t, graph_dict_t):
        """Calculate GAN loss for discriminator D"""

        _, _, _, _, latent_z_s, _ = self.netG(x_s.to(self.device), graph_dict_s['adj_norm'].to(self.device))
        loss_dis_s = self.criterionGAN(self.netD(latent_z_s), True)
        _, _, _, _, latent_z_t, _ = self.netG(x_t.to(self.device), graph_dict_t['adj_norm'].to(self.device))
        loss_dis_t = self.criterionGAN(self.netD(latent_z_t), False)
        self.loss_DIS = (loss_dis_s + loss_dis_t) * 0.5
        self.loss_DIS.backward()

    def backward_G_S(self, x_s, l_s, graph_dict_s):
        """Calculate the loss for generators G and C by source domain."""
        lambda_F = self.opt.lambda_F  # weight of AE branch
        lambda_G = self.opt.lambda_G  # weight of VGAE branch
        lambda_C = self.opt.lambda_C  # weight of CLS branch

        # Rec loss and Cls loss
        gcn_mu_s, gcn_logstd_s, _, _, latent_z_s, decoded_x_s = self.netG(x_s.to(self.device), graph_dict_s['adj_norm'].to(self.device))
        loss_gcn_s = self.criterionREC_G(preds=self.netG.dc(latent_z_s), labels=graph_dict_s['adj_label'].to(self.device), mu=gcn_mu_s, logstd=gcn_logstd_s, 
                                         n_nodes=graph_dict_s['adj_label'].shape[0], norm=graph_dict_s['norm_value'])
        loss_rec_s = self.criterionREC_F(decoded_x_s, x_s.to(self.device))
        loss_cls_s = self.criterionCLS(self.netC(latent_z_s), l_s.long().to(self.device))

        self.loss_REC = lambda_F * loss_rec_s + lambda_G * loss_gcn_s
        self.loss_CLS = lambda_C * loss_cls_s
        self.loss_G = self.loss_REC + self.loss_CLS
        self.loss_G.backward()

    def backward_G_T(self, x_t, graph_dict_t):
        """Calculate the loss for generators G and C by source domain."""
        lambda_F = self.opt.lambda_F  # weight of AE branch
        lambda_G = self.opt.lambda_G  # weight of VGAE branch
        lambda_A = self.opt.lambda_A  # weight of GAN branch

        # Rec loss and GAN loss
        gcn_mu_t, gcn_logstd_t, _, _, latent_z_t, decoded_x_t = self.netG(x_t.to(self.device), graph_dict_t['adj_norm'].to(self.device))
        loss_gcn_t = self.criterionREC_G(preds=self.netG.dc(latent_z_t), labels=graph_dict_t['adj_label'].to(self.device), mu=gcn_mu_t, logstd=gcn_logstd_t, 
                                         n_nodes=graph_dict_t['adj_label'].shape[0], norm=graph_dict_t['norm_value'])
        loss_rec_t = self.criterionREC_F(decoded_x_t, x_t.to(self.device))
        loss_gan_t = self.criterionGAN(self.netD(latent_z_t), True)

        self.loss_GAN = lambda_A * loss_gan_t
        self.loss_A = self.loss_GAN + lambda_F * loss_rec_t + lambda_G * loss_gcn_t
        self.loss_A.backward()


    def optimize_parameters(self, x_s, l_s, graph_dict_s, x_t, graph_dict_t):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.netG.train()
        self.netC.train()

        # Step 1. Training G and C with source domain.
        self.set_requires_grad([self.netG, self.netC], True)       # optimizing G and C.
        self.set_requires_grad([self.netD], False)                 # D require no gradients when optimizing G.
        self.optimizer_G.zero_grad()                               # set G gradients to zero.
        self.optimizer_C.zero_grad()                               # set G gradients to zero.
        self.backward_G_S(x_s, l_s, graph_dict_s)                  # calculate gradients for G and C.
        self.optimizer_G.step()                                    # update weights of G.
        self.optimizer_C.step()                                    # update weights of C.

        # Step 2. Training D.
        self.set_requires_grad([self.netG, self.netC], False)      # G_S and G_T require no gradients when optimizing D_S and D_T.
        self.set_requires_grad([self.netD], True)                  # optimizing D_S and D_T.
        self.optimizer_D.zero_grad()                               # set D_S and D_T gradients to zero.
        self.backward_D(x_s, graph_dict_s, x_t, graph_dict_t)      # calculate gradients for D_S.
        self.optimizer_D.step()                                    # update D_S and D_T weights.

        # Step 3. GAN training G with target domain.
        self.set_requires_grad([self.netG], True)                  # optimizing G.
        self.set_requires_grad([self.netD], False)                 # D require no gradients when optimizing G.
        self.optimizer_G.zero_grad()                               # set G gradients to zero.
        self.backward_G_T(x_t, graph_dict_t)                       # calculate gradients for G and C.
        self.optimizer_G.step()                                    # update weights of G.


    def inference(self, x_t, graph_dict_t):
        """Make inference models to eval mode and no_grad() during test time, so we don't save intermediate steps for backprop"""
        self.netG.eval()
        self.netC.eval()
        with torch.no_grad():
            _, _, _, _, latent_z_t, decoded_x_t = self.netG(x_t.to(self.device), graph_dict_t['adj_norm'].to(self.device))
            outputs = self.netC(latent_z_t)
        return F.softmax(outputs, dim=-1)


    def setup(self, opt):
        """Load and print networks
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.load_epoch)
        self.print_networks(opt.verbose)

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                    torch.save(net.module.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
