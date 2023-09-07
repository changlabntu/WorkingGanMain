# XBM code from https://github.com/msight-tech/research-xbm/tree/master

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
import numpy as np
from models.helper_oai import OaiSubjects, classify_easy_3d, swap_by_labels
from models.helper import reshape_3d, tile_like
from utils.contrastive_loss import triplet_loss, SupConLoss, ContrastiveLoss, XbmTripletLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from utils.cross_batch_memory import XBM


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feature_shapes):
        for mlp_id, feat in enumerate(feature_shapes):
            input_nc = feat
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # if len(self.gpu_ids) > 0:
            # mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        # print(len(feats))
        # print([x.shape for x in feats])
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            # B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            # (B, C, H, W)
            # feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            feat = feat.permute(0, 2, 3, 1)  # (B, H*W, C)
            feat_reshape = feat.view(feat.shape[0], feat.shape[1] * feat.shape[2], feat.shape[3])
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    # patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])  # (random order of range(H*W))
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device) # first N patches
                    # patch_id = torch.from_numpy(patch_id).type(torch.long).to(feat.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)  # Channel (1, 128, 256, 256, 256) > (256, 256, 256, 256, 256)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        # print([x.shape for x in return_feats]) # (B * num_patches, 256) * level of features
        return return_feats, return_ids


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        self.hparams.final = 'none'
        self.net_g, self.net_d = self.set_networks()
        self.hparams.final = 'none'
        self.net_gY, _ = self.set_networks()
        self.classifier = nn.Conv2d(256, 2, 1, stride=1, padding=0).cuda()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g', 'net_gY': 'net_gY', 'netF': 'netF'}
        self.netd_names = {'net_d': 'net_d', 'classifier': 'classifier'}

        self.oai = OaiSubjects(self.hparams.dataset)

        if hparams.lbvgg > 0:
            self.VGGloss = VGGLoss().cuda()

        # CUT NCE
        self.featDown = nn.MaxPool2d(kernel_size=self.hparams.fDown)  # extra pooling to increase field of view

        netF = PatchSampleF(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[], nc=256)
        self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
        feature_shapes = [32, 64, 128, 256]
        self.netF.create_mlp(feature_shapes)

        if self.hparams.fWhich == None:  # which layer of the feature map to be considered in CUT
            self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

        print(self.hparams.fWhich)

        self.criterionNCE = []
        for nce_layer in range(4):  # self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device))

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()
        self.XBM = XBM(hparams)
        self.iter = 0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--xbm', action='store_true', help='use xbm')
        parser.add_argument('--lbX', type=float, default=0, help='weight for xbm loss')
        parser.add_argument('--xbm_size', type=int, default=100, help='size of xbm')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--start_iter', type=int, default=0, help='save xbms after this many iterations')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--fDown", dest='fDown', type=int, default=1)
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        parser.add_argument("--adv", dest='adv', type=float, default=1)
        parser.add_argument("--lbvgg", dest='lbvgg', type=float, default=0)
        parser.add_argument("--alpha", dest='alpha', type=int,
                            help='ending epoch for decaying skip connection, 0 for no decaying', default=0)
        return parent_parser

    @staticmethod
    def test_method(net_g, img, hparams, a=None):
        oriX = img[0]

        imgXY = net_g(oriX, alpha=0)
        if not hparams.nomask:
            imgXY = nn.Sigmoid()(imgXY)  # mask
            combined = combine(imgXY, oriX, method='mul')
        else:
            combined = imgXY

        return {'imgXY': imgXY[0, ::].detach().cpu(), 'combinedXY': combined[0, ::].detach().cpu()}

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.iter += 1
        if optimizer_idx == 0:
            self.generation(batch)
            loss_d = self.backward_d()
            if loss_d is not None:
                for k in list(loss_d.keys()):
                    if k != 'sum':
                        self.log(k, loss_d[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                return loss_d['sum']
            else:
                return None

        if optimizer_idx == 1:
            self.generation(batch)  # why there are two generation?
            self.xbm = self.hparams.xbm
            loss_g = self.backward_g()
            for k in list(loss_g.keys()):
                if k != 'sum':
                    self.log(k, loss_g[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return loss_g['sum']

    def validation_step(self, batch, batch_idx):  # 定義Validation如何進行，以這邊為例就再加上了計算Acc.
        self.generation(batch)
        self.xbm = False
        loss_g = self.backward_g()
        loss = loss_g['loss_xbm']
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': loss}


    def validation_epoch_end(self, outputs):  # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def generation(self, batch):
        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])
        self.xbm = self.hparams.xbm

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
        self.outXz = self.net_g(self.oriX, alpha=1, method='encode')
        outX = self.net_g(self.outXz, alpha=1, method='decode')
        self.imgXY = nn.Sigmoid()(outX['out0'])  # mask 0 - 1
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')  # i am using masking (0-1) here

        #
        self.outYz = self.net_g(self.oriY, alpha=alpha, method='encode')
        outY = self.net_gY(self.outYz, alpha=alpha, method='decode')
        self.imgYY = nn.Tanh()(outY['out0'])  # -1 ~ 1, real img

        # pain label (its not in used for now)
        self.labels = self.oai.labels_unilateral(filenames=batch['filenames'])

    def backward_g(self):
        # ADV(XY)+
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=True)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY)

        loss_l1Y = self.add_loss_l1(a=self.imgYY, b=self.oriY)

        loss_ga = axy  # * 0.5 + axx * 0.5

        loss_g = loss_ga * self.hparams.adv + loss_l1 * self.hparams.lamb + loss_l1Y * self.hparams.lamb

        # CUT NCE_loss
        if self.hparams.lbNCE > 0:
            # (Y, YY) (XY, YY) (Y, XY)
            feat_q = self.net_g(self.oriY, method='encode')
            feat_k = self.net_g(self.imgYY, method='encode')

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

        if self.hparams.lbX > 0:
            crit = ContrastiveLoss()
            featDown = nn.MaxPool2d(self.outXz[-1].shape[-1])
            pool = nn.MaxPool1d(23)
            # pool = nn.AvgPool1d(23)
            imgs = [self.outXz[-1], self.outYz[-1]]
            feats = [featDown(f) for f in imgs]
            feats = [f.view(f.shape[0] // 23, f.shape[1], 23) for f in feats]
            feats = [pool(f) for f in feats] #(b, 256, 1)
            feats = torch.cat(feats, 0).squeeze()

            labels = torch.zeros(feats.shape[0]).to(self.device)
            labels[feats.shape[0]//2:] = 1

            loss_ori = crit(feats, labels, feats, labels)
            if (self.iter >= self.hparams.start_iter) and self.xbm:
                self.XBM.enqueue_dequeue(feats.detach(), labels.detach())
                xbm_feats, xbm_labels = self.XBM.get()
                xbm = crit(feats, labels, xbm_feats, xbm_labels)
                loss_xbm = loss_ori + xbm
            else:
                loss_xbm = loss_ori

        else:
            loss_xbm = 0

        loss_g += loss_nce * self.hparams.lbNCE + loss_xbm * self.hparams.lbX

        return {'sum': loss_g, 'l1': loss_l1, 'ga': loss_ga, 'loss_nce': loss_nce, 'loss_xbm': loss_xbm}

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


# python train.py --prj mlp/test/ --models lesion_cutGB --jsn lesion_cut --env t09b --nm 01 --fDown 4 --use_mlp --fWhich 0 0 0 1
