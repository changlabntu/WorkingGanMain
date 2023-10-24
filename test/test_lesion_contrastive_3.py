import torch
import numpy as np
import glob, os, sys, time
import tifffile as tiff
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.data_utils import imagesc
import umap
reducer = umap.UMAP()
from os import path
import sys
from collections import OrderedDict

#sys.path.append(path.abspath('../WorkingGan'))
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

def convert_linear_to_conv2d(model):
    # convert Linear to Conv2d
    new_model = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias
            new_layer = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1,
                                  padding=0, bias=True)
            new_layer.weight.data = module.weight.data.view(out_features, in_features, 1, 1)
            if bias is not None:
                new_layer.bias.data = bias.data
            new_model.append((name, new_layer))
        elif isinstance(module, nn.ReLU):
            new_model.append((name, module))
    new_model = nn.Sequential(OrderedDict(new_model))
    return new_model

def get_tiff_stack(x):
    x = x / x.max()
    if nm11:
        x = (x - 0.5) * 2
    x = torch.from_numpy(x).unsqueeze(1).float().cuda()
    return x


# get images and segmentation
def get_images_and_seg(list_img):
    x0 = get_tiff_stack(np.stack([tiff.imread(la[x]) for x in list_img], 0))
    x1 = get_tiff_stack(np.stack([tiff.imread(la[x].replace('/a/', '/b/')) for x in list_img], 0))
    seg0 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/aseg/')) for x in list_img], 0)
    eff0 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/aeff/')) for x in list_img], 0)
    seg1 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/bseg/')) for x in list_img], 0)
    eff1 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/beff/')) for x in list_img], 0)
    return x0, x1, seg0, eff0, seg1, eff1


def get_seg(x):
    # get segmentation, but apply maxpooling (8 * fDown) to match the size of the feature map
    x = torch.from_numpy(x)
    x = nn.MaxPool2d(8 * fDown)(x / 1)
    x = x.permute(1, 2, 0).reshape(-1)
    return x


def get_model(prj_name, epoch, option='new'):
    if option == 'new':
        #model = torch.load('/media/ExtHDD01/logs/womac4/0719/alpha0_cutGB_vgg10_nce1/checkpoints/net_g_model_epoch_120.pth',
        #                   map_location=torch.device('cpu')).cuda()

        model = torch.load(
            '/media/ExtHDD01/logs/womac4/' + prj_name + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
            map_location=torch.device('cpu')).cuda()
        #model = torch.load(
        #    '/media/ExtHDD01/logs/womac4/mlpb2/ngf16fD2fW0011b4/checkpoints/net_g_model_epoch_100.pth',
        #    map_location=torch.device('cpu')).cuda()

    elif option == 'old':
        model = torch.load('/media/ExtHDD01/logs/womac4/3D/test4fixmcVgg10/checkpoints/net_g_model_epoch_40.pth',
                           map_location=torch.device('cpu')).cuda(); nomask = False; nm11 = False;

    try:
        netF = torch.load('/media/ExtHDD01/logs/womac4/' + prj_name +
                          '/checkpoints/netF_model_epoch_' + str(epoch) + '.pth', map_location=torch.device('cpu')).cuda()
        l2norm = netF.l2norm
        netF = [convert_linear_to_conv2d(x) for x in [netF.mlp_0, netF.mlp_1, netF.mlp_2, netF.mlp_3]]
    except:
        netF = None
        l2norm = None

    return model, netF, l2norm


def pain_significance_monte_carlo(x0, x1, model, skip=1, nomask=False):
    # pain significance
    outall = []
    print('running monte carlo.....')
    for mc in range(100):
        if nomask:
            try:
                o = model(x0, alpha=skip)['out0'][:, 0, :, :].detach().cpu()
            except:
                o = model(x0, a=1)['out0'][:, 0, :, :].detach().cpu()
        else:
            try:
                o = model(x0, alpha=1, method='encode')
                o = model(o, alpha=1, method='decode')['out0'].detach().cpu()
            except:
                o = model(x0, a=1)['out0'].detach().cpu()
            o = nn.Sigmoid()(o)
            o = torch.multiply(o[:, 0, :, :], x0[:, 0, :, :].cpu())
        outall.append(o)
    print('done running monte carlo.....')

    outall = [x0[:, 0, :, :].cpu() - x for x in outall]

    mean = torch.mean(torch.stack(outall, 3), 3)
    var = torch.var(torch.stack(outall, 3), 3)
    sig = torch.divide(mean, torch.sqrt(var) + 0.01)
    return mean, var, sig

###
# Prepare data and model
###
# get the model
#prj_name = 'mlp/test0/' # good
#prj_name = 'mlp/alpha0_cutGB2_vgg0_nce4_0001/'
#prj_name = 'mlpb2/ngf24fD2fW0001b4/'
#prj_name = 'mlp/cutGB2_nce1_1111_ngf16_patch512/'
#prj_name = 'GB_2b/down2_nce10_0001_ngf32_patch512_crop128/'
prj_name = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16/'


model, netF, l2norm = get_model(prj_name, epoch=200)
print([type(x) for x in [model, netF, l2norm]])

nomask = False
nm11 = False
fDown = 2  # the
skip = 1
fWhich = [0, 0, 1, 1]  # which layers of features to use

root = '/media/ExtHDD01/Dataset/paired_images/womac4/full/'
output_path = '/home/ghc/Dataset/paired_images/womac4contrastive/'
# list of images
la = sorted(glob.glob(root + 'ap/*'))

# list of images to be tested
#list_img = [41, 534, 696, 800, 827, 1180, 1224, 1290, 6910, 9256] #+ list(range(8))
#list_img = [x - 1 for x in list_img]  # -1 because imagej is 1-indexed
for sub in range(2225):
    tini = time.time()
    #list_img = list(range(0, 46))

    list_img = list(range(sub*23, (sub+1)*23))

    # name of the images
    name = [la[y].split('/')[-1] for y in list_img]

    # load images and segmentations
    x0, x1, seg0, eff0, seg1, eff1 = get_images_and_seg(list_img=list_img)
    #(seg0, eff0, seg1, eff1) = [get_seg(x) for x in (seg0, eff0, seg1, eff1)]

    ###
    # get pain significance by monte carlo
    ###

    mean, var, sig = pain_significance_monte_carlo(x0, x1, model, skip=skip, nomask=nomask)

    for folder_name in ['sig', 'mean', 'var', 'eff', 'seg', 'fmap', 'ori'][:3]:
        os.makedirs(output_path + folder_name, exist_ok=True)

    for i in range(len(list_img)):
        tiff.imwrite(output_path + 'sig/' + name[i], sig[i, :, :].cpu().numpy())
        tiff.imwrite(output_path + 'mean/' + name[i], mean[i, :, :].cpu().numpy())
        tiff.imwrite(output_path + 'var/' + name[i], var[i, :, :].cpu().numpy())

    print(time.time() - tini)