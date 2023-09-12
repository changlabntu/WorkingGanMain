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
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from models.lesion_cutGB_2a import PatchSampleF
from models.lesion_global1 import TripletCenterLoss, CenterLoss


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        self.hparams.final = 'none'
        self.net_g, self.net_d = self.set_networks()
        self.hparams.final = 'none'
        self.net_gY, _ = self.set_networks()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g', 'net_gY': 'net_gY', 'netF': 'netF'}
        self.netd_names = {'net_d': 'net_d'}

        self.oai = OaiSubjects(self.hparams.dataset)

        if hparams.lbvgg > 0:
            self.VGGloss = VGGLoss().cuda()

        # CUT NCE
        self.featDown = nn.MaxPool2d(kernel_size=self.hparams.fDown)  # extra pooling to increase field of view

        netF = PatchSampleF(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[],
                            nc=self.hparams.c_mlp)
        self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
        feature_shapes = [x * self.hparams.ngf for x in [1, 2, 4, 8]]
        self.netF.create_mlp(feature_shapes)

        if self.hparams.fWhich == None:  # which layer of the feature map to be considered in CUT
            self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

        print(self.hparams.fWhich)

        self.criterionNCE = []
        for nce_layer in range(4):  # self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device))

        if 0:
            # global contrastive
            self.batch = self.hparams.batch_size
            self.pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
            self.triple = nn.TripletMarginLoss()
            if self.hparams.projection > 0:
                self.center = CenterLoss(feat_dim=self.hparams.projection)
            else:
                self.center = CenterLoss(feat_dim=512)
            self.tripletcenter = TripletCenterLoss()

            if self.hparams.projection > 0:
                self.net_g.projection = nn.Linear(512, self.hparams.projection).cuda()

            # Finally, initialize the optimizers and scheduler
            self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument("--projection", dest='projection', type=int, default=0)
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument("--fDown", dest='fDown', type=int, default=1)
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        parser.add_argument("--adv", dest='adv', type=float, default=1)
        parser.add_argument("--lbvgg", dest='lbvgg', type=float, default=0)
        parser.add_argument("--alpha", dest='alpha', type=int,
                            help='ending epoch for decaying skip connection, 0 for no decaying', default=0)
        return parent_parser

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        self.oriX = batch['img'][0]
        self.oriY = batch['img'][1]

        # decaying skip connection
        if self.hparams.alpha > 0:  # if decaying
            alpha = 1 - self.epoch / self.hparams.alpha
            if alpha < 0:
                alpha = 0
        else:
            alpha = 0  # if always disconnected

        # generating a mask by sigmoid to locate the lesions, turn out its the best way for now
        outXz = self.net_g(self.oriX, alpha=1, method='encode')
        outX = self.net_g(outXz, alpha=1, method='decode')
        self.imgXY = nn.Sigmoid()(outX['out0'])  # mask 0 - 1
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')  # i am using masking (0-1) here

        #
        outYz = self.net_g(self.oriY, alpha=alpha, method='encode')
        outY = self.net_gY(outYz, alpha=alpha, method='decode')
        self.imgYY = nn.Sigmoid()(outY['out0'])  # -1 ~ 1, real img

        # global contrastive
        # use last layer
        if 0:
            self.outXz = outXz[-1]
            self.outYz = outYz[-1]

            (B, C, X, Y) = self.outXz.shape
            self.outXz = self.outXz.view(B//self.batch, self.batch, C, X, Y)
            self.outXz = self.outXz.permute(1, 2, 3, 4, 0)
            self.outXz = self.pool(self.outXz)[:, :, 0, 0, 0]
            if self.hparams.projection > 0:
                self.outXz = self.net_g.projection(self.outXz)

            outYz = self.net_g(self.oriY, alpha=alpha, method='encode')
            self.outYz = self.outYz.view(B//self.batch, self.batch, C, X, Y)
            self.outYz = self.outYz.permute(1, 2, 3, 4, 0)
            self.outYz = self.pool(self.outYz)[:, :, 0, 0, 0]
            if self.hparams.projection > 0:
                self.outYz = self.net_g.projection(self.outYz)

    def backward_g(self):
        loss_dict = dict()

        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=True)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY)

        loss_l1Y = self.add_loss_l1(a=self.imgYY, b=self.oriY)

        loss_ga = axy  # * 0.5 + axx * 0.5

        loss_g = loss_ga * self.hparams.adv + loss_l1 * self.hparams.lamb + loss_l1Y * self.hparams.lamb

        if self.hparams.lbvgg > 0:
            loss_gvgg = self.VGGloss(torch.cat([self.imgXY] * 3, 1), torch.cat([self.oriY] * 3, 1))
            loss_g += loss_gvgg * self.hparams.lbvgg
        else:
            loss_gvgg = 0

        # CUT NCE_loss
        if self.hparams.lbNCE > 0:
            # (Y, YY) (XY, YY) (Y, XY)
            feat_q = self.net_g(self.oriY, method='encode')
            feat_k = self.net_g(self.imgXY, method='encode')

            feat_q = [self.featDown(f) for f in feat_q]
            feat_k = [self.featDown(f) for f in feat_k]

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                                None)  # get source patches by random id
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
        else:
            loss_nce = 0

        loss_g += loss_nce * self.hparams.lbNCE

        loss_dict['loss_nce'] = loss_nce
        loss_dict['loss_l1'] = loss_l1
        loss_dict['loss_l1Y'] = loss_l1Y

        # global contrastive
        if 0:
            loss_t = 0
            loss_t += self.triple(self.outYz[:1, ::], self.outYz[1:, ::], self.outXz[:1, ::])
            loss_t += self.triple(self.outYz[1:, ::], self.outYz[:1, ::], self.outXz[1:, ::])
            loss_center = self.center(torch.cat([f for f in [self.outXz, self.outYz]], dim=0), torch.FloatTensor([0, 0, 1, 1]).cuda())
            loss_g += loss_t + loss_center

            loss_dict['loss_t'] = loss_t
            loss_dict['loss_center'] = loss_center

        loss_dict['sum'] = loss_g
        return loss_dict

    def cut_sample(self, feat_q, feat_k):
        feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                            None)  # get source patches by random id
        feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

        total_nce_loss = 0.0
        for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
            loss = crit(f_q, f_k) * f_w
            total_nce_loss += loss.mean()
        return total_nce_loss

    def backward_d(self):
        # ADV(XY)-
        axy = self.add_loss_adv(a=self.imgXY.detach(), net_d=self.net_d, truth=False)

        # ADV(XX)-
        # axx, _ = self.add_loss_adv_classify3d(a=self.imgXX, net_d=self.net_dX, truth_adv=False, truth_classify=False)
        ay = self.add_loss_adv(a=self.oriY.detach(), net_d=self.net_d, truth=True)

        # adversarial of xy (-) and y (+)
        loss_da = axy * 0.5 + ay * 0.5  # axy * 0.25 + axx * 0.25 + ax * 0.25 + ay * 0.25
        # classify x (+) vs y (-)
        loss_d = loss_da * self.hparams.adv

        return {'sum': loss_d, 'da': loss_da}


# CUDA_VISIBLE_DEVICES=0 python train.py --alpha 0 --jsn womac4 --projection global/0/ --models lesion_global --netG edalphand --split a --dataset womac4 --lbvgg 0 -b 2 --nm 01
