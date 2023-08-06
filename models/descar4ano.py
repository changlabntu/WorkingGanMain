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
        #self.net_dX = copy.deepcopy(self.net_d)
        self.classifier = nn.Conv2d(256, 1, 1, stride=1, padding=0).cuda()

        self.net_z = copy.deepcopy(self.net_d)

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g', 'net_z': 'net_z', 'classifier': 'classifier'}
        self.netd_names = {'net_d': 'net_d'}#, 'net_class': 'netDC'}

        self.oai = OaiSubjects(self.hparams.dataset)

        if hparams.lbvgg > 0:
            self.VGGloss = VGGLoss().cuda()

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lbx", dest='lbx', type=float, default=1)
        parser.add_argument("--dc0", dest='dc0', type=float, default=1)
        parser.add_argument("--adv", dest='adv', type=float, default=1)
        parser.add_argument("--fix", dest='fix', action='store_true', default=False)
        parser.add_argument("--lbvgg", dest='lbvgg', type=float, default=10)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, a=None):
        oriX = img[0]

        print(a)
        imgXX = net_g(oriX, a=torch.FloatTensor([0]))
        imgXX = nn.Sigmoid()(imgXX['out0'])  # mask

        imgXY = net_g(oriX, a=torch.FloatTensor([a]))
        imgXY = nn.Sigmoid()(imgXY['out0'])  # mask

        imgXX = combine(imgXX, oriX, method='mul')
        imgXY = combine(imgXY, oriX, method='mul')

        return imgXY

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        # pain label
        self.labels = self.oai.labels_unilateral(filenames=batch['filenames'])
        self.oriX = batch['img'][0]
        self.oriY = batch['img'][1]

        if self.hparams.fix:
            a = torch.ones_like(self.labels['paindiff'])
        else:
            a = torch.abs(self.labels['paindiff'])

        outXa = self.net_g(self.oriX, a=a)
        self.imgXY = nn.Sigmoid()(outXa['out0'])  # mask
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')

        outX0 = self.net_g(self.oriX, a=0 * torch.abs(self.labels['paindiff']))
        self.imgXX = nn.Sigmoid()(outX0['out0'])  # mask
        self.imgXX = combine(self.imgXX, self.oriX, method='mul')

        imgXYz = self.net_g(self.imgXY, a=0 * torch.abs(self.labels['paindiff']))['z']
        oriYz = self.net_g(self.oriY, a=0 * torch.abs(self.labels['paindiff']))['z']

        #self.oriXz = outXa['z']
        self.imgXYz = imgXYz
        self.oriYz = oriYz

    def backward_g(self):
        # ADV(XY)+
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=True)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY)

        # L1(XX, X)
        loss_l1x = self.add_loss_l1(a=self.imgXX, b=self.oriX)

        loss_ga = axy# * 0.5 + axx * 0.5

        # Z
        loss_z = self.MSELoss(self.oriYz, self.imgXYz) * 1000

        loss_g = loss_ga + loss_l1 * self.hparams.lamb + loss_l1x * self.hparams.lbx\
                 + loss_z

        if self.hparams.lbvgg > 0:
            loss_gvgg = self.VGGloss(torch.cat([self.imgXY] * 3, 1), torch.cat([self.oriY] * 3, 1))
            loss_g += loss_gvgg * self.hparams.lbvgg

        return {'sum': loss_g, 'l1': loss_l1, 'ga': loss_ga, 'z': loss_z}

    def backward_d(self):
        # ADV(XY)-
        axy = self.add_loss_adv(a=self.imgXY.detach(), net_d=self.net_d, truth=False)

        # ADV(XX)-
        ay = self.add_loss_adv(a=self.oriY.detach(), net_d=self.net_d, truth=True)

        loss_da = axy * 0.5 + ay * 0.5#axy * 0.25 + axx * 0.25 + ax * 0.25 + ay * 0.25
        # classify x (+) vs y (-)
        loss_d = loss_da * self.hparams.adv

        return {'sum': loss_d, 'da': loss_da}




#A CUDA_VISIBLE_DEVICES=0,1 python train.py --jsn womac3 --prj 3D/test4ano/0/  --models descar4ano --netG dsnumcrel0a --netD bpatch_16 --split moaks