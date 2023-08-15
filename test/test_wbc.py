import torch
import os, glob
from utils.data_utils import imagesc
from PIL import Image, ImageFilter
import numpy as np
import torch.nn as nn

def test_wbc_cyc():
    net = torch.load('/media/ExtHDD01/logs/WBC1/cyc_wbc/unet128/checkpoints/net_gXY_model_epoch_100.pth',map_location='cpu').cuda()
    #net = torch.load('/media/ExtHDD01/logs/WBC/srgan/scale4/checkpoints/net_g_model_epoch_200.pth').cuda()
    up = nn.Upsample(size=(256, 256), mode='bicubic', align_corners=False)

    l = sorted(glob.glob('/home/ghc/Dataset/paired_images/WBC1/full/wbc/myelo/*'))

    x0 = np.array(Image.open(l[84]))
    xori = 1 * x0
    x0 = x0/x0.max()
    x0 = (x0 - 0.5) * 2

    x0 = torch.from_numpy(x0.astype(np.float32)).per mute(2, 0, 1).unsqueeze(0).cuda()
    x0 = up(x0)
    out = net(x0)['out0']
    imagesc(x0[0,:,:,:].detach().cpu())
    imagesc(out[0,:,:,:].detach().cpu())

def test_wbc_cyc_mask():
    net = torch.load('/media/ExtHDD01/logs/WBC1/cyc_wbc/unet128_mask/checkpoints/netGXY_model_epoch_300.pth',
                     map_location='cpu').cuda()
    # net = torch.load('/media/ExtHDD01/logs/WBC/srgan/scale4/checkpoints/net_g_model_epoch_200.pth').cuda()
    up = nn.Upsample(size=(256, 256), mode='bicubic', align_corners=False)

    l = sorted(glob.glob('/home/ghc/Dataset/paired_images/WBC1/full/wbc/myelo/*'))

    x0 = np.array(Image.open(l[47]))
    m0 = np.array(Image.open(l[47].replace('/wbc/', '/mask/')))
    xori = 1 * x0
    x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2
    x0 = torch.from_numpy(x0.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda()
    x0 = up(x0)

    m0 = m0 / m0.max()
    m0 = (m0 - 0.5) * 2
    m0 = torch.from_numpy(m0.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    m0 = up(m0).repeat(1, 3, 1, 1)


    out = net(torch.cat([x0, m0], 1), a=0)
    imagesc(x0[0, :, :, :].detach().cpu())
    imagesc(out['out0'][0, :, :, :].detach().cpu())
    imagesc(out['out1'][0, :, :, :].detach().cpu())
