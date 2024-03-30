import torch
import numpy as np
import glob, os, sys
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
    for mc in range(50):
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

def plot_style_1():
    for condition in [0, 1, 2]:
        le = (lesion >= 0) / 1
        plt.scatter(e[le == 1, 0], e[le == 1, 1], s=0.05 * np.ones(((le == 1).sum(), 1))[:, 0])
        for trd in [4, 6, 8]:
            if condition == 0:
                le = (lesion >= trd) & (seg > 0) & (pain == P)
            if condition == 1:
                le = (lesion >= trd) & (eff == 1) & (pain == P)
            if condition == 2:
                le = (lesion >= trd) & (eff == 0) & (seg == 0) & (pain == P)
            if condition == 3:
                le = (lesion >= trd)

            plt.scatter(e[le == 1, 0], e[le == 1, 1], s=2 * np.ones(((le == 1).sum(), 1))[:, 0])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

###
# Prepare data and model
###
# get the model
#prj_name = 'mlp/test0/' # good
#prj_name = 'mlp/alpha0_cutGB2_vgg0_nce4_0001/'
#prj_name = 'mlpb2/ngf24fD2fW0001b4/'
#prj_name = 'mlp/cutGB2_nce1_1111_ngf16_patch512/'
#prj_name = 'GB_2b/down2_nce10_0001_ngf32_patch512_crop128/'
prj_name = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16_unpaired_moaks/'

model, netF, l2norm = get_model(prj_name, epoch=160)
print([type(x) for x in [model, netF, l2norm]])

nomask = False
nm11 = False
fDown = 1  # the
skip = 1
fWhich = [0, 0, 1, 1]  # which layers of features to use

root = '/media/ExtHDD01/Dataset/paired_images/womac4/full/'
# list of images
la = sorted(glob.glob(root + 'a/*'))

# list of images to be tested
list_img = [41, 534, 696, 800, 827, 1180, 1224, 1290, 6910, 9256] #+ list(range(8))
#list_img = list(range(41, 51))
list_img = [x - 1 for x in list_img]  # -1 because its 1-indexed
#list_img = list_img + [x + 10 for x in list_img]

# name of the images
name = [la[y].split('/')[-1] for y in list_img]

# load images and segmentations
x0, x1, seg0, eff0, seg1, eff1 = get_images_and_seg(list_img=list_img)
(seg0, eff0, seg1, eff1) = [get_seg(x) for x in (seg0, eff0, seg1, eff1)]

###
# get pain significance by monte carlo
###

mean, var, sig = pain_significance_monte_carlo(x0, x1, model, skip=skip, nomask=nomask)

# collect features
f0 = model(x0, method='encode')
f1 = model(x1, method='encode')

if netF is not None:
    f0 = [l2norm(m(f)) for m, f in zip(netF, f0)]
    f1 = [l2norm(m(f)) for m, f in zip(netF, f1)]

f0 = [x.cpu().detach() for x in f0]
f1 = [x.cpu().detach() for x in f1]

f0 = [nn.MaxPool2d(fDown * j)(i) for (i, j) in zip(f0, [8, 4, 2, 1])]
f1 = [nn.MaxPool2d(fDown * j)(i) for (i, j) in zip(f1, [8, 4, 2, 1])]

# combine all the layers, by fWhich
f0 = [x for (x, y) in zip(f0, fWhich) if y == 1]
f1 = [x for (x, y) in zip(f1, fWhich) if y == 1]
f0 = torch.cat(f0, 1)
f1 = torch.cat(f1, 1)

C = f0.shape[1]
f0 = f0.permute(1, 2, 3, 0).reshape(C, -1).numpy()
f1 = f1.permute(1, 2, 3, 0).reshape(C, -1).numpy()

# features
# data = f0[-1].permute(1, 2, 3, 0).reshape(256, -1).cpu().detach().numpy()
# e0 = tsne.fit_transform(data.T)

# lesion
lesion = nn.MaxPool2d(8 * fDown)(sig)
lesion = lesion.permute(1, 2, 0).reshape(-1)

# double everything
lesion = np.concatenate([lesion, lesion * 0], 0)
seg = np.concatenate([seg0, seg1], 0)
eff = np.concatenate([eff0, eff1], 0)

# features
data = np.concatenate([f0, f1], 1)
import time

tini = time.time()
e = reducer.fit_transform(data.T)   # umap
print('umap time used: ' + str(time.time() - tini))
e[:, 0] = (e[:, 0] - e[:, 0].min()) / (e[:, 0].max() - e[:, 0].min())
e[:, 1] = (e[:, 1] - e[:, 1].min()) / (e[:, 1].max() - e[:, 1].min())

# pain
pain = np.concatenate([np.ones((f0.shape[1])), 2 * np.ones((f1.shape[1]))])
P = 1


#plt.figure(figsize=(7, 7))
trd = 4
# Is lesion
le = (lesion >= 0) / 1
plt.scatter(e[le == 1, 0], e[le == 1, 1], s=1 * np.ones(((le == 1).sum(), 1))[:, 0], alpha=0.2)
# Is lesion and in Seg
le = (lesion >= trd) & (seg > 0) & (pain == P)
plt.scatter(e[le == 1, 0], e[le == 1, 1], s=10 * np.ones(((le == 1).sum(), 1))[:, 0], alpha=0.2)
# Is lesion and in Eff
le = (lesion >= trd) & (eff == 1) & (pain == P)
plt.scatter(e[le == 1, 0], e[le == 1, 1], s=10 * np.ones(((le == 1).sum(), 1))[:, 0], alpha=0.2)
#le = (lesion >= trd) & (eff == 0) & (seg == 0) & (pain == P)
#plt.scatter(e[le == 1, 0], e[le == 1, 1], s=10 * np.ones(((le == 1).sum(), 1))[:, 0], alpha=0.2)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

#plt.figure(figsize=(7, 7))
trd = 4
le = (lesion >= 0) / 1
plt.scatter(e[le == 1, 0], e[le == 1, 1], s=1 * np.ones(((le == 1).sum(), 1))[:, 0], alpha=0.2)
le = (lesion >= trd) & (eff == 0) & (seg == 0) & (pain == P)
plt.scatter(e[le == 1, 0], e[le == 1, 1], s=10 * np.ones(((le == 1).sum(), 1))[:, 0], alpha=0.2)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()


label = -10 * np.ones((e.shape[0]))

#label[lesion >= 8] = 1
# label[lesion < 4] = 0

label[(lesion >= 4) & (seg > 0)] = 1
label[(lesion >= 4) & (eff > 0)] = 1
# label[lesion < 2] = 0
label[(pain == 2)] = 0

# knn
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(e[label >= 0, :], label[label >= 0])
fout = knn.predict(e)

# plot for no pain
fout[label >= 0] = label[label >= 0]
plt.scatter(e[fout == 0, 0], e[fout == 0, 1], s=0.05 * np.ones(((fout == 0).sum(), 1))[:, 0])
plt.scatter(e[fout == 1, 0], e[fout == 1, 1], s=2 * np.ones(((fout == 1).sum(), 1))[:, 0])
plt.show()

# resize back to pixel space
fout = fout[:fout.shape[0] // 2]
fout = fout.reshape((48 // fDown, 48 // fDown, len(list_img)))
fout = torch.from_numpy(fout).permute(2, 0, 1).unsqueeze(1)

for folder_name in ['fmap', 'ori', 'sig', 'mean', 'eff', 'seg']:
    os.makedirs('output/' + folder_name, exist_ok=True)

x0, x1, seg0, eff0, seg1, eff1 = get_images_and_seg(list_img=list_img)
for i in range(len(list_img)):
    #tiff.imwrite('output/ori/' + name[i], x0[i, 0, :, :].cpu().numpy())
    #tiff.imwrite('output/seg/' + name[i], seg0[i, :, :])
    tiff.imwrite('output/eff/' + name[i], eff0[i, :, :])
    tiff.imwrite('output/fmap/' + name[i], fout[i, 0, :, :].numpy().astype(np.float32))
    tiff.imwrite('output/sig/' + name[i], sig[i, :, :].cpu().numpy())
    tiff.imwrite('output/mean/' + name[i], mean[i, :, :].cpu().numpy())

fout = nn.Upsample(scale_factor=8 * fDown, mode='bilinear')(fout)
fout[fout <= 0] = 0

# sig = torch.from_numpy(tiff.imread('sigold.tif'))
fout = torch.multiply(fout[:, 0, :, :], sig).numpy()

tiff.imwrite('output/' + prj_name.split('/')[1] + '.tif', np.concatenate([fout, sig.numpy()], 2))

# pain-no pain comparison
# plt.scatter(e[:e.shape[0] // 2, 0], e[:e.shape[0] // 2, 1], s=0.05 * np.ones(e.shape[0] // 2))
# plt.scatter(e[e.shape[0] // 2:, 0], e[e.shape[0] // 2:, 1], s=0.01 * np.ones(e.shape[0] // 2))
# plt.show()