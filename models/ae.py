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


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        self.net_g, self.net_d = self.set_networks()
        self.net_gY = copy.deepcopy(self.net_g)

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g', 'net_gY': 'net_gY'}
        self.netd_names = {'net_d': 'net_d'}

        self.oai = OaiSubjects(self.hparams.dataset)

        if hparams.lbvgg > 0:
            self.VGGloss = VGGLoss().cuda()

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lbvgg", dest='lbvgg', type=float, default=0)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, args, a=None):
        print(a)
        oriX = img[0]

        imgXY = net_g(oriX)['out0']

        combined = imgXY

        return {'imgXY': imgXY[0, ::].detach().cpu(), 'combinedXY': combined[0, ::].detach().cpu()}

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        # pain label
        #self.labels = self.oai.labels_unilateral(filenames=batch['filenames'])
        self.oriX = batch['img'][0]
        self.oriY = batch['img'][1]

        outX = self.net_g(self.oriX)
        self.imgXY = outX['out0']

        outY = self.net_gY(self.oriY)
        self.imgYY = outY['out0']

    def backward_g(self):
        # L2(XY, Y)
        loss_l2X = self.criterionL2(self.imgXY, self.oriY)
        loss_l2Y = self.criterionL2(self.imgYY, self.oriY)

        loss_g = loss_l2X * self.hparams.lamb + loss_l2Y * self.hparams.lamb

        return {'sum': loss_g, 'l2': loss_l2X + loss_l2Y}#, 'gvgg': loss_gvgg}

    def backward_d(self):
        #loss_d = 0 * self.add_loss_adv(a=self.imgXY.detach(), net_d=self.net_d, truth=False)
        return None


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn womac4 --prj 0623/descar5/0/  --models descar5 --netG dsnumc --dataset womac4 --split all --final tanh --nm 11