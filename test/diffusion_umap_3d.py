import torch
import glob, os, sys
import tifffile as tiff
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
reducer = umap.UMAP()
import sys, os
#sys.path.append(path.abspath('../WorkingGan'))
import torch.nn as nn
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
    # subject_path = sorted(glob.glob(root + folder + id +'_015.tif'))
    # cube = tiff.imread(subject_path[0])
    subject_path = sorted(glob.glob(root + folder + id + '*.tif'))
    cube = np.stack([tiff.imread(x) for x in subject_path], 0)
    return cube

def get_cube_images(list_img):
    x0 = get_tiff_stack(np.stack([get_cube(x, 'a/') for x in list_img])) # torch.Size([10, 1, 23, 384, 384])
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

def get_feat(x):
    # collect features
    f0 = model(x, method='encode')
    # use last layer feature only
    f0 = f0[-1:][0]
    featDown = nn.MaxPool2d(kernel_size=48)
    f0 = featDown(f0)
    # sub = f0.shape[0]
    # f0 = f0.permute(1, 2, 3, 0).reshape(-1, sub).cpu().detach().numpy()  # (2304, 10)
    f0 = f0.flatten().unsqueeze(1).cpu().detach().numpy()
    return f0

def get_model(prj):
    model = torch.load(
        f'/media/ziyi/glory/logs_pin/womac4/{prj}/checkpoints/net_g_model_epoch_60.pth',
        map_location=torch.device('cpu')).cuda()
    model.eval()
    return model

def get_dist(x1, x2, label=None):
    if label:
        x1 = x1[labels_super == 0]
        x2 = x2[labels_super == label]
    dist = x1 - x2
    dist_value = np.mean([np.linalg.norm(dist[i]) for i in range(x1.shape[0])])
    dist_xy = np.mean(dist, axis=0)
    return dist_value, dist_xy
###
# Prepare data and model
###
nomask = False
nm11 = False
# prj = '0823_triploss_nobnmc_flat'
prj = '0825_triploss_flat4_nobn'
root = '/media/ziyi/glory/OAIDataBase/womac4/full/'
# get the model
model = get_model(prj = prj)
# list of images
la = sorted(glob.glob(root + 'a/*'))

# list of images to be tested
import random
random.seed(21)
list_img = random.sample(range(0, 667*23), 50)
# name of the images
name = [la[y].split('/')[-1] for y in list_img]
subject_name = [x[:-8] for x in name]
subjects = list(dict.fromkeys([x.split('/')[-1] for x in subject_name]))

# load images and segmentations
x0, x1 = get_cube_images(subjects)

super_diff, normal_diff = [], []
for folder in ['ori', 'mess', 'eff', 'all']:
    super_diff.append(get_diffusion('superpain', folder))
for folder in ['ori', 'mess', 'eff', 'all']:
    normal_diff.append(get_diffusion('nopain', folder))
base = [x0, x1]
base = torch.cat(base, 0) #torch.Size([96, 23, 384, 384])
super = torch.cat(super_diff, 0)
normal = torch.cat(normal_diff, 0)
all = torch.cat([base, super, normal], 0)

all_f = []
for i in range(all.shape[0]): #run by subject
    x = all[i:i+1,:,:,:].permute(1, 0, 2, 3) #torch.Size([23, 1, 384, 384])
    f = get_feat(x)
    all_f.append(f)

#     # collect features
# base_f, super_f = [], []
# for i in range(23): #run by 2d
#     f = get_feat(all[:, i:i+1,:, :])
#     f = np.expand_dims(f, axis=1)
#     all_f.append(f)

# for i in range(base.shape[0]): #run by subject
#     x = base[i:i+1,:,:,:].permute(1, 0, 2, 3)
#     f = get_feat(x)
#     base_f.append(f)
# for i in range(super.shape[0]): #run by subject
#     x = super[i:i+1,:,:,:].permute(1, 0, 2, 3)
#     f = get_feat(x)
#     super_f.append(f)

# base_f = np.concatenate(base_f, 1).reshape(256*23,-1) #(256*23, 96)
# print(base_f.shape)
# super_f = np.concatenate(super_f, 1).reshape(256*23,-1) #(256*23, 8*4)
# data = np.concatenate([base_f, super_f], 1)
# data = base_f
# base_f = np.concatenate(all_f, 1).reshape(256*23,-1) #(256*23, 96)
data = np.concatenate(all_f, 1).reshape(256*23,-1)
print(data.shape)

import time
tini = time.time()
e = reducer.fit_transform(data.T)   # umap (20, 2)
print(e.shape)
print('umap time used: ' + str(time.time() - tini))
e[:, 0] = (e[:, 0] - e[:, 0].min()) / (e[:, 0].max() - e[:, 0].min())
e[:, 1] = (e[:, 1] - e[:, 1].min()) / (e[:, 1].max() - e[:, 1].min())

sub = int(len(subjects))
labels_base = np.zeros(2*sub)
labels_base[sub:2*sub] = 1
# set labes for d
labels_super = np.zeros(8*4)
labels_super[8:16] = 1
labels_super[16:24] = 2
labels_super[-8:] = 3
# plot base
plt.scatter(e[:2*sub, 0], e[:2*sub, 1], c=labels_base, cmap='RdYlBu', alpha=0.5, marker="o")
# plot super
plt.scatter(e[2*sub:2*sub+32, 0], e[2*sub:2*sub+32, 1], c=labels_super, cmap='Spectral', marker="^")
# plot normal
plt.scatter(e[2*sub+32:, 0], e[2*sub+32:, 1], c=labels_super, cmap='Spectral', marker="s")

# caculate the distance
base_dist_value, base_dist_xy = get_dist(e[:sub, :], e[sub:2*sub, :])
super_dist_value, super_dist_xy = get_dist(e[2*sub:2*sub+32, :], e[2*sub:2*sub+32, :], 3)
super_mess_dist_value, super_mess_dist_xy = get_dist(e[2*sub:2*sub+32, :], e[2*sub:2*sub+32, :], 1)
super_eff_dist_value, super_eff_dist_xy = get_dist(e[2*sub:2*sub+32, :], e[2*sub:2*sub+32, :], 2)
normal_dist_value, normal_dist_xy = get_dist(e[-32:, :], e[-32:, :], 3)

print('base:', base_dist_value, base_dist_xy)
print('super:', super_dist_value, super_dist_xy)
print('super_mess:', super_mess_dist_value, super_mess_dist_xy)
print('super_eff:', super_eff_dist_value, super_eff_dist_xy)
print('normal:', normal_dist_value, normal_dist_xy)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.figtext(0.2, 0.9, 'Red pain - Blue nopain - Green eff - Yellow mess')
plt.figtext(0.2, 0.03, 'Base: circle, Super: triangle, Normal: square')
# plt.show()
os.makedirs('./output/umap/lesion', exist_ok=True)
plt.savefig(f'./output/umap/lesion/{prj}.png')

d = {'name': ['base', 'super', 'super_mess', 'super_eff', 'normal'],
     'dist_value': [base_dist_value, super_dist_value, super_mess_dist_value, super_eff_dist_value, normal_dist_value],
     'dist_xy': [base_dist_xy, super_dist_xy, super_mess_dist_xy, super_eff_dist_xy, normal_dist_xy]}
df = pd.DataFrame(data=d)
df.to_csv(f'./output/umap/lesion/{prj}.csv', index=False)