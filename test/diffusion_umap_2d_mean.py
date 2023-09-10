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
    x = torch.from_numpy(x).float().cuda()
    return x

def get_cube(id, folder):
    subject_path = sorted(glob.glob(root + folder + id +'*.tif'))
    cube = np.stack([tiff.imread(x) for x in subject_path], 0)
    return cube

# get images and segmentation
def get_cube_images(list_img):
    x0 = get_tiff_stack(np.stack([get_cube(x, 'a/') for x in list_img])) # torch.Size([10, 23, 384, 384])
    x1 = get_tiff_stack(np.stack([get_cube(x, 'b/') for x in list_img]))
    return x0, x1

def get_diffusion(pain_level,name):
    files = sorted(glob.glob(root + f'0821_Diffusion_res/{pain_level}/{name}/*'))
    sub_num = int(len(files)/23)
    ls = []
    for i in range(sub_num):
        x = np.stack([tiff.imread(x) for x in files[i*23:(i+1)*23]], 1)
        x = x.transpose(1,0,2)
        ls.append(x)
    x = np.stack(ls, 0)
    x = get_tiff_stack(x)
    return x

def get_model():
    model = torch.load(
        '/media/ziyi/glory/logs_pin/womac4/0904_xbm2/checkpoints/net_g_model_epoch_150.pth',
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
list_img = random.sample(range(0, 667*23), 30)
# name of the images
name = [la[y].split('/')[-1] for y in list_img]
subject_name = [x[:-8] for x in name]
subjects = list(dict.fromkeys([x.split('/')[-1] for x in subject_name]))

# load images and segmentations
x0, x1 = get_cube_images(subjects)
super_diff, normal_diff, no_diff = [], [], []
for folder in ['ori', 'mess', 'eff', 'all']:
    super_diff.append(get_diffusion('superpain', folder))
# for folder in ['ori', 'mess', 'eff', 'all']:
#     normal_diff.append(get_diffusion('normalpain', folder))
# for folder in ['ori', 'mess', 'eff', 'all']:
#     no_diff.append(get_diffusion('nopain', folder))
base = torch.cat([x0, x1], 0) #torch.Size([96, 23, 384, 384])
super = torch.cat(super_diff, 0)
# normal = torch.cat(normal_diff, 0)
# no = torch.cat(no_diff, 0)
all = torch.cat([base, super], 0)

f_ls = []
for i in range(23):
    # collect feature
    f = model(all[:, i:i+1,:, :], method='encode')

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

import time
tini = time.time()
e = reducer.fit_transform(data.T)   # umap (20, 2)
print(e.shape)
print('umap time used: ' + str(time.time() - tini))
e[:, 0] = (e[:, 0] - e[:, 0].min()) / (e[:, 0].max() - e[:, 0].min())
e[:, 1] = (e[:, 1] - e[:, 1].min()) / (e[:, 1].max() - e[:, 1].min())

mean_ls = []
for i in range(sub*2+32):
    e_mean = np.mean(e[i*23:(i+1)*23, :],0)
    e_mean = np.expand_dims(e_mean, 0)
    mean_ls.append(e_mean)
e = np.concatenate(mean_ls, 0)
print(e.shape)

labels = np.zeros(e.shape[0])
labels[sub:2*sub] = 3
labels[2*sub:2*sub+8] = 1
labels[2*sub+8:2*sub+16] = 5
labels[2*sub+16:2*sub+24] = 2
labels[2*sub+24:2*sub+32] = 4

plt.scatter(e[:, 0], e[:, 1], c=labels, cmap='RdYlBu')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.figtext(0.4, 0.9, 'Red [p ,p8, eff, bl, np, all] Blue')
plt.show()
# os.makedirs('./output/umap/lesion', exist_ok=True)
# plt.savefig(f'./output/umap/lesion/{args.test}.png')
