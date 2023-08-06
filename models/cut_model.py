import torch, copy
import torch.nn as nn
import torchvision
import torch.optim as optim
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *
from models.base import BaseModel, combine, VGGLoss
import pandas as pd
from models.helper_oai import OaiSubjects, classify_easy_3d, swap_by_labels
from models.helper import reshape_3d, tile_like
import utils.util_cut as util
import networks.networks_cut as networks
from networks.networks_cut import PatchNCELoss


class GAN(BaseModel):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        # from CUT base_options
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
        parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')

        parser.set_defaults(pool_size=0)  # no image pooling
        return parent_parser

    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        self.automatic_optimization = False
        self.isTrain = True
        self.gpu_ids = []
        self.opt = self.hparams
        self.opt.isTrain = self.isTrain
        opt = self.opt

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        pytorch_total_params = sum(p.numel() for p in self.netG.parameters() if p.requires_grad)
        print("Total number of trainable parameters in G: ", pytorch_total_params)

        #self.optimizer_f = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
        #                                    betas=(self.opt.beta1, 0.999))

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode)#.to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt))#.to(self.device))

            self.criterionIdt = torch.nn.L1Loss()#.to(self.device)
            #self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            #self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            #self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)

        self.data_dependent_initialize()

        # update names of the models for optimization
        self.netg_names = {'netG': 'netG', 'netF': 'netF'}
        self.netd_names = {'netD': 'netD'}

        # Finally, initialize the optimizers and scheduler
        self.init_optimizer_scheduler()

    def data_dependent_initialize(self):
        # [torch.Size([16, 1, 262, 262]), torch.Size([16, 128, 256, 256]), torch.Size([16, 256, 128, 128]), torch.Size([16, 256, 64, 64]), torch.Size([16, 256, 64, 64])]
        feature_shapes = [(16, self.opt.input_nc, 262, 262), (16, 128, 256, 256), (16, 256, 128, 128), (16, 256, 64, 64), (16, 256, 64, 64)]
        feats = [torch.rand(x) for x in feature_shapes]
        self.netF.create_mlp(feats)
        self.optimizer_f = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def data_dependent_initializeX(self):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        print('data_dependent_initialize begin')
        for i, x in enumerate(self.train_loader):
            if i < 2: continue
            batch = x
        #batch['img'] = [x.cuda() for x in batch['img']]
        self.generation(batch)                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_d()['sum'].backward()                  # calculate gradients for D
            self.backward_g()['sum'].backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_f = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                #self.optimizers.append(self.optimizer_F)
        print('data_dependent_initialize end')


    @staticmethod
    def test_method(net_g, img, args, a=None):
        return None#{'imgXY': imgXY[0, ::].detach().cpu(), 'combinedXY': combined[0, ::].detach().cpu()}

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        # pain label
        self.real_A = batch['img'][1]   # INVERTED FOR NOW!!!
        self.real_B = batch['img'][0]

        #print((self.real_A.min(), self.real_A.max()))

        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        #tiff.imwrite('real.tif', torch.cat((self.real_A[0, 0, :, :], self.real_B[0, 0, :, :]), dim=1).cpu().numpy())

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def backward_d(self):
        """Calculate GAN loss for the discriminator"""
        #fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        return {'sum': self.loss_D, 'd_real': self.loss_D_real, 'd_fake': self.loss_D_fake}

    def backward_g(self):
        """Calculate GAN and NCE loss for the generator"""
        #fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        #print((self.loss_G_GAN, loss_NCE_both))

        return {'sum': self.loss_G, 'g_gan': self.loss_G_GAN, 'nce': loss_NCE_both}#, 'gvgg': loss_gvgg}

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)  # get source patches by random id
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)  # use the ids for query target

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def training_step(self, batch):
        # forward
        self.generation(batch)

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_d.zero_grad()
        self.loss_d = self.backward_d()
        self.loss_d['sum'].backward()
        self.optimizer_d.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_g.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_f.zero_grad()
        self.loss_g = self.backward_g()
        self.loss_g['sum'].backward()
        self.optimizer_g.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_f.step()

        for k in list(self.loss_d.keys()):
            if k is not 'sum':
                self.log(k, self.loss_d[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        for k in list(self.loss_g.keys()):
            if k is not 'sum':
                self.log(k, self.loss_g[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

# CUDA_VISIBLE_DEVICES=0 python train.py --jsn cut_cat --prj 4/  --models cut_model --netG resnet_9blocks --netD basic --split all --env t09b --lambda_NCE 1 -b 1
# CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn cut_oai --prj cyc_cut/  --models cut_model --netG resnet_9blocks --netD basic --split all --env t09b --lambda_NCE 1 -b 1