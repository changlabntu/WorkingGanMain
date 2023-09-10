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
    subject_path = sorted(glob.glob(root + folder + id + '*.tif'))
    cube = np.stack([tiff.imread(x) for x in subject_path], 0)
    return cube

def get_cube_images(list_img):
    x0 = get_tiff_stack(np.stack([get_cube(x, 'a/') for x in list_img])) # torch.Size([10, 1, 23, 384, 384])
    x1 = get_tiff_stack(np.stack([get_cube(x, 'b/') for x in list_img]))
    x0 = x0.squeeze()
    x1 = x1.squeeze()
    return x0, x1

def get_model():
    model = torch.load(
        '/media/ziyi/glory/logs_pin/womac4/0906_xbm_max_256/checkpoints/net_g_model_epoch_20.pth',
        map_location=torch.device('cpu')).cuda()
    # model.eval()
    return model

###
# Prepare data and model
###

nm11 = False

root = '/media/ziyi/glory/OAIDataBase/womac4/full/'
# list of images
la = sorted(glob.glob(root + 'a/*'))

# list of images to be tested
import random
# random.seed(37)
list_img = random.sample(range(0, 667*23), 50)
# list_img = random.sample(range(667*23, 23000), 50)
# print(list_img)
# list_img = [41, 534, 696, 800, 827, 1180, 1224, 1290, 6910, 9256]
# list_img = [x - 1 for x in list_img]  # -1 because its 1-indexed

# name of the images
name = [la[y].split('/')[-1] for y in list_img]

# pick 10 subjects
subject_name = [x[:-8] for x in name]
subjects = list(dict.fromkeys([x.split('/')[-1] for x in subject_name]))
print(len(subjects),' subjects')

# load images
x0, x1 = get_cube_images(subjects)
x = torch.cat([x0, x1], 0)

f_ls = []
# get the model
model = get_model()

for i in range(x.shape[0]):
    # collect feature
    f = model(x[i:i+1,: ,: , :].permute(1,0,2,3), method='encode')
    # use last layer feature only
    f = f[-1:][0]
    featDown = nn.MaxPool2d(kernel_size=48)
    max_pool = nn.MaxPool1d(23)
    f = featDown(f)
    f = f.view(1, f.shape[1], 23)
    f = max_pool(f).squeeze(2)
    f = f.cpu().detach().numpy()
    f_ls.append(f)

f = np.concatenate(f_ls, 0)
sub = len(subjects)
print(f.shape)

import time
tini = time.time()
e = reducer.fit_transform(f)   # umap (20, 2)
print(e.shape)
print('umap time used: ' + str(time.time() - tini))
e[:, 0] = (e[:, 0] - e[:, 0].min()) / (e[:, 0].max() - e[:, 0].min())
e[:, 1] = (e[:, 1] - e[:, 1].min()) / (e[:, 1].max() - e[:, 1].min())

plt.scatter(e[:sub, 0], e[:sub, 1])
plt.scatter(e[sub:, 0], e[sub:, 1])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# os.makedirs('./output/umap/3d', exist_ok=True)
# plt.savefig(f'./output/umap/3d/{args.test}.png')
