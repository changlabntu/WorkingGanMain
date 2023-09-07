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
    # subject_path = sorted(glob.glob(root + folder + id +'_015.tif'))
    # cube = tiff.imread(subject_path[0])
    subject_path = sorted(glob.glob(root + folder + id + '*.tif'))
    cube = np.stack([tiff.imread(x) for x in subject_path], 0)
    return cube

def get_cube_images(list_img):
    x0 = get_tiff_stack(np.stack([get_cube(x, 'a/') for x in list_img])) # torch.Size([10, 1, 23, 384, 384])
    x1 = get_tiff_stack(np.stack([get_cube(x, 'b/') for x in list_img]))
    x0 = x0.squeeze()
    x1 = x1.squeeze()
    return x0, x1

# get images and segmentation
def get_images_and_seg(list_img):
    x0 = get_tiff_stack(np.stack([tiff.imread(la[x]) for x in list_img], 1))
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
    # 0831_supcon2d ep180
    model = torch.load(
        '/media/ziyi/glory/logs_pin/womac4/0904_xbm2/checkpoints/net_g_model_epoch_150.pth',
        map_location=torch.device('cpu')).cuda()
    # model.eval()
    return model


###
# Prepare data and model
###

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
# list_img = random.sample(range(0, 667*23), 20)
list_img = random.sample(range(667*23, 23000), 40)
# name of the images
name = [la[y].split('/')[-1] for y in list_img]
subject_name = [x[:-8] for x in name]
subjects = list(dict.fromkeys([x.split('/')[-1] for x in subject_name]))

# load images
x0, x1 = get_cube_images(subjects)
x = torch.cat([x0, x1], 0)
print(x.shape)

model = get_model()

f_ls = []
# for i in range(23):
#     # collect feature
#     f = model(x[:, i:i+1,:, :], method='encode')
#
#     # use last layer feature only
#     f = f[-1:]
#     f = torch.cat(f, 1)
#     featDown = nn.MaxPool2d(kernel_size=48)
#     f = featDown(f)
#     f = f.squeeze(3).cpu().detach().numpy()
#     f_ls.append(f)

for i in range(x.shape[0]):
    # collect feature
    f = model(x[i:i+1,:, :, :].permute(1,0,2,3), method='encode')
    # use last layer feature only
    f = f[-1:]
    f = torch.cat(f, 1)
    featDown = nn.MaxPool2d(kernel_size=48)
    f = featDown(f)
    f = f.squeeze(3).cpu().detach().numpy()
    f_ls.append(f)

data = np.concatenate(f_ls, 2)
data = data.reshape(data.shape[1], -1)
print(data.shape)
sub = len(subjects)
# features

import time
tini = time.time()
e = reducer.fit_transform(data.T)   # umap (20, 2)
print('umap time used: ' + str(time.time() - tini))
print(e.shape)
e[:, 0] = (e[:, 0] - e[:, 0].min()) / (e[:, 0].max() - e[:, 0].min())
e[:, 1] = (e[:, 1] - e[:, 1].min()) / (e[:, 1].max() - e[:, 1].min())

# mean_ls = []
# for i in range(sub*2):
#     e_mean = np.mean(e[i*23:(i+1)*23, :],0)
#     e_mean = np.expand_dims(e_mean, 0)
#     mean_ls.append(e_mean)
# e = np.concatenate(mean_ls, 0)
# #
# plt.scatter(e[:sub, 0], e[:sub, 1])
# plt.scatter(e[sub:, 0], e[sub:, 1])
plt.scatter(e[:sub*23, 0], e[:sub*23, 1])
plt.scatter(e[sub*23:, 0], e[sub*23:, 1])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
# os.makedirs('./output/umap/2d', exist_ok=True)
# plt.savefig(f'./output/umap/2d/{args.test}.png')
