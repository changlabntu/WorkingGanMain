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

def get_seg(x):
    # get segmentation, but apply maxpooling (8 * ratio) to match the size of the feature map
    x = torch.from_numpy(x)
    x = nn.MaxPool2d(8 * ratio)(x / 1)
    x = x.permute(1, 2, 0).reshape(-1)
    return x


def get_model():
    model = torch.load(
        '/media/ziyi/glory/logs_pin/womac4/0904_supcon2d_mcbn/checkpoints/net_g_model_epoch_150.pth',
        map_location=torch.device('cpu')).cuda()
    model.eval()

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
# random.seed(21)
list_img = random.sample(range(0, 667*23), 50)
# print(list_img)
# list_img = [41, 534, 696, 800, 827, 1180, 1224, 1290, 6910, 9256]
# list_img = [x - 1 for x in list_img]  # -1 because its 1-indexed

# name of the images
name = [la[y].split('/')[-1] for y in list_img]

# pick 10 subjects
subject_name = [x[:-8] for x in name]

# load images and segmentations
x0, x1, seg0, eff0, seg1, eff1 = get_images_and_seg(list_img)
x = torch.cat([x0, x1], 0)
print(x.shape)
# collect features
f = model(x, method='encode')
# use last layer feature only
f = f[-1:][0]
featDown = nn.MaxPool2d(kernel_size=48)
f = featDown(f) #torch.Size([10, 256, 3, 3])

sub = f.shape[0]
C = f.shape[1]
f = f.permute(1, 2, 3, 0).reshape(-1, sub).cpu().detach().numpy() #(2304, 10)
print(f.shape)

import time
tini = time.time()
e = reducer.fit_transform(f.T)   # umap (20, 2)
print(e.shape)
print('umap time used: ' + str(time.time() - tini))
e[:, 0] = (e[:, 0] - e[:, 0].min()) / (e[:, 0].max() - e[:, 0].min())
e[:, 1] = (e[:, 1] - e[:, 1].min()) / (e[:, 1].max() - e[:, 1].min())

cut = int(sub / 2)
plt.scatter(e[:cut, 0], e[:cut, 1])
plt.scatter(e[cut:, 0], e[cut:, 1])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
# os.makedirs('./output/umap/2d', exist_ok=True)
# plt.savefig(f'./output/umap/2d/{args.test}.png')
