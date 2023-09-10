import torch
import numpy as np
import glob, os, sys
import tifffile as tiff
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from utils.data_utils import imagesc
import umap
reducer = umap.UMAP()
from os import path
import sys, os

#sys.path.append(path.abspath('../WorkingGan'))
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', type=str)
args = parser.parse_args()

root_path = os.getcwd()
sys.path.append(root_path)

def get_tiff_stack(x):
    x = x / x.max()
    if nm11:
        x = (x - 0.5) * 2
    x = torch.from_numpy(x).unsqueeze(1).float().cuda()
    return x

def get_cube(id, folder):
    subject_path = sorted(glob.glob(root + folder + id +'*.tif'))
    cube = np.stack([tiff.imread(x) for x in subject_path], 0)
    return cube

# get images and segmentation
def get_images_and_seg(list_img):
    x0 = get_tiff_stack(np.stack([tiff.imread(la[x]) for x in list_img], 0))
    x1 = get_tiff_stack(np.stack([tiff.imread(la[x].replace('/a/', '/b/')) for x in list_img], 0))
    seg0 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/aseg/')) for x in list_img], 0)
    eff0 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/aeff/')) for x in list_img], 0)
    seg1 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/bseg/')) for x in list_img], 0)
    eff1 = np.stack([tiff.imread(la[x].replace('/a/', '/seg/beff/')) for x in list_img], 0)
    return x0, x1, seg0, eff0, seg1, eff1


def get_feat(x):
    # collect features
    f0 = model(x, method='encode')
    # use last layer feature only
    f0 = f0[-1:]
    f0 = torch.cat(f0, 1)
    featDown = nn.MaxPool2d(kernel_size=48)
    f0 = featDown(f0)  # torch.Size([10, 256, 3, 3])
    sub = f0.shape[0]
    f0 = f0.permute(1, 2, 3, 0).reshape(-1, sub).cpu().detach().numpy()  # (2304, 10)
    return f0

def get_model(option='new'):
    if option == 'new':
        model = torch.load(
            '/media/ziyi/glory/logs_pin/womac4/0821_triploss/checkpoints/net_g_model_epoch_100.pth',
            map_location=torch.device('cpu')).cuda()
        # model.eval()

    return model


###
# Prepare data and model
###
# get the model
model = get_model()

nomask = False
nm11 = False
ratio = 1  # the
skip = 1
fWhich = [0, 0, 0, 1]  # which layers of features to use

root = '/media/ziyi/glory/OAIDataBase/womac4/full/'
# list of images
la = sorted(glob.glob(root + 'a/*'))

# list of images to be tested
import random
random.seed(21)
list_img = random.sample(range(0, 667*23), 40)

# load images and segmentations
x0, x1, seg0, eff0, seg1, eff1 = get_images_and_seg(list_img)

ori_path = glob.glob(root + 'Diffusion_res/ori/*')
dif_path = glob.glob(root + 'Diffusion_res/all/*')
eff_path = glob.glob(root + 'Diffusion_res/eff/*')
mess_path = glob.glob(root + 'Diffusion_res/mess/*')
ori = get_tiff_stack(np.stack([tiff.imread(x) for x in ori_path], 0))
dif = get_tiff_stack(np.stack([tiff.imread(x) for x in dif_path], 0))
eff = get_tiff_stack(np.stack([tiff.imread(x) for x in eff_path], 0))
mess = get_tiff_stack(np.stack([tiff.imread(x) for x in mess_path], 0))
all = torch.cat([x0, x1, ori, dif, eff, mess], 0)

data = get_feat(all)
import time
tini = time.time()
e = reducer.fit_transform(data.T)   # umap (20, 2)
print(e.shape)
print('umap time used: ' + str(time.time() - tini))
e[:, 0] = (e[:, 0] - e[:, 0].min()) / (e[:, 0].max() - e[:, 0].min())
e[:, 1] = (e[:, 1] - e[:, 1].min()) / (e[:, 1].max() - e[:, 1].min())

sub = len(list_img)
labels = np.zeros(e.shape[0])
labels[sub:2*sub] = 3
labels[2*sub:2*sub+8] = 1
labels[2*sub+8:2*sub+16] = 5
labels[2*sub+16:2*sub+24] = 2
labels[-8:] = 4

plt.scatter(e[:, 0], e[:, 1], c=labels, cmap='RdYlBu')
# plt.scatter(e[:sub, 0], e[:sub, 1])
# plt.scatter(e[sub:2*sub, 0], e[sub:2*sub, 1])
# plt.scatter(e[2*sub:2*sub+8, 0], e[2*sub:2*sub+8, 1])
# plt.scatter(e[-8:, 0], e[-8:, 1])

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.figtext(0.4, 0.9, 'Red [p ,p8, eff, bl, np, all] Blue')
plt.show()
# os.makedirs('./output/umap/lesion', exist_ok=True)
# plt.savefig(f'./output/umap/lesion/{args.test}.png')
