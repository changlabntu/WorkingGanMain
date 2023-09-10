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
# sys.path.append(path.abspath('../WorkingGan'))
root_path = os.getcwd()
sys.path.append(root_path)
import torch.nn as nn
from models.helper_oai import OaiSubjects
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', type=str)
args = parser.parse_args()

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
    f = model(x, method='encode')
    # use last layer feature only
    f = f[-1:][0]
    featDown = nn.MaxPool2d(kernel_size=48)
    max_pool = nn.MaxPool1d(23)
    f = featDown(f)
    f = f.view(f.shape[0] // 23, f.shape[1], 23)
    f = max_pool(f).squeeze(2).cpu().detach().numpy()
    # f0 = f0.flatten().unsqueeze(1).cpu().detach().numpy()
    return f

def get_model(prj):
    model = torch.load(
        f'/media/ziyi/glory/logs_pin/womac4/{prj}/checkpoints/net_g_model_epoch_40.pth',
        map_location=torch.device('cpu')).cuda()
    model.eval()
    return model

def get_dist(x1, x2, label_ls=None, label=None):
    if label:
        x1 = x1[label_ls == 0]
        x2 = x2[label_ls == label]
    dist = x1 - x2
    dist_value = np.mean([np.linalg.norm(dist[i]) for i in range(x1.shape[0])])
    dist_xy = np.mean(dist, axis=0)
    return dist_value, dist_xy

def get_dist_center(x, labels_ls=None, label=None):
    dist = []
    if label:
        for i in label:
            x0 = x[labels_ls == i]
            nopain_dist = [np.linalg.norm(x0[j] - center_nopain) for j in range(x0.shape[0])]
            mean_dist = np.mean(nopain_dist)
            dist.append(mean_dist)
    else:
        dist = [np.linalg.norm(x[j] - center_nopain) for j in range(x.shape[0])]
    return dist

###
# Prepare data and model
###
nomask = False
nm11 = False
prj = '0906_xbm_max_256'
# prj = '0825_triploss_flat4_nobn'
print(prj)
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
# filenames = [la[y] for y in list_img]
name = [la[y].split('/')[-1] for y in list_img]
subject_name = [x[:-8] for x in name]
subjects = list(dict.fromkeys([x.split('/')[-1] for x in subject_name]))

# get pain labels
oai = OaiSubjects("womac4")
# print([x[:-3] for x in subject_name])
labels = oai.labels_unilateral(filenames=subjects, mode='test')

# load images and segmentations
x0, x1 = get_cube_images(subjects)

super_diff, normal_diff, no_diff = [], [], []
for folder in ['ori', 'mess', 'eff', 'all']:
    super_diff.append(get_diffusion('superpain', folder))
for folder in ['ori', 'mess', 'eff', 'all']:
    normal_diff.append(get_diffusion('normalpain', folder))
for folder in ['ori', 'mess', 'eff', 'all']:
    no_diff.append(get_diffusion('nopain', folder))
base = torch.cat([x0, x1], 0) #torch.Size([96, 23, 384, 384])
super = torch.cat(super_diff, 0)
normal = torch.cat(normal_diff, 0)
no = torch.cat(no_diff, 0)
all = torch.cat([base, super, normal, no], 0)
print(len(all))

all_f = []
for i in range(all.shape[0]): #run by subject
    x = all[i:i+1,:,:,:].permute(1, 0, 2, 3) #torch.Size([23, 1, 384, 384])
    f = get_feat(x)
    all_f.append(f)

data = np.concatenate(all_f, 0)
print(data.shape)

import time
tini = time.time()
# e = reducer.fit_transform(data.T)   # umap (20, 2)
e = TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(data)
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
labels_no = np.zeros(3*4)
labels_no[3:6] = 1
labels_no[6:9] = 2
labels_no[-3:] = 3

# calculate center
print('distance to unpain center:')
center_nopain = np.mean(e[sub:2*sub, :], axis=0)
# pain_dist = get_dist_center(e[:sub, :])
pain_dist = [np.linalg.norm(e[j:j+1, :] - e[sub+j:sub+j+1, :]) for j in range(sub)]
# print('pain:', pain_dist)
# print('pain score:', labels['paindiff'])
womac_x = [abs(int(x*10)) for x in labels['paindiff']]
womac_y = [round(x, 2) for x in pain_dist]
coef = np.polyfit(womac_x, womac_y,1)
print(coef)
poly1d_fn = np.poly1d(coef)
plt.plot(womac_x, womac_y, 'yo', womac_x, poly1d_fn(womac_x), '--k')
# plt.show()
plt.savefig(f'./output/umap/lesion/{prj}_womac_pair.png')
plt.close()
super_cn = get_dist_center(e[2*sub:2*sub+32, :], labels_super, [0,3])
normal_cn = get_dist_center(e[2*sub+32:2*sub+64, :], labels_super, [0,3])
no_cn = get_dist_center(e[-12:, :], labels_no, [0,3])
print('super:', super_cn)
print('normal:', normal_cn)
print('no:', no_cn)

# caculate the distance
base_dist_value, base_dist_xy = get_dist(e[:sub, :], e[sub:2*sub, :])
super_dist_value, super_dist_xy = get_dist(e[2*sub:2*sub+32, :], e[2*sub:2*sub+32, :], labels_super, 3)
super_mess_dist_value, super_mess_dist_xy = get_dist(e[2*sub:2*sub+32, :], e[2*sub:2*sub+32, :], labels_super, 1)
super_eff_dist_value, super_eff_dist_xy = get_dist(e[2*sub:2*sub+32, :], e[2*sub:2*sub+32, :], labels_super, 2)
normal_dist_value, normal_dist_xy = get_dist(e[2*sub+32:2*sub+64, :], e[2*sub+32:2*sub+64, :], labels_super, 3)
no_dist_value, no_dist_xy = get_dist(e[-12:, :], e[-12:, :], labels_no, 3)
print('distance from pain to unpain:')
print('base:', base_dist_value, base_dist_xy)
print('super:', super_dist_value, super_dist_xy)
print('super_mess:', super_mess_dist_value, super_mess_dist_xy)
print('super_eff:', super_eff_dist_value, super_eff_dist_xy)
print('normal:', normal_dist_value, normal_dist_xy)
print('no:', no_dist_value, no_dist_xy)

# plot base
plt.scatter(e[:2*sub, 0], e[:2*sub, 1], c=labels_base, cmap='RdYlBu', alpha=0.5, marker="o")
# plot super
plt.scatter(e[2*sub:2*sub+32, 0], e[2*sub:2*sub+32, 1], c=labels_super, cmap='Spectral', marker="^")
# plot normal
plt.scatter(e[2*sub+32:2*sub+64, 0], e[2*sub+32:2*sub+64, 1], c=labels_super, cmap='Spectral', marker="s")
# plot normal
plt.scatter(e[-12:, 0], e[-12:, 1], c=labels_no, cmap='Spectral', marker="*")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.figtext(0.2, 0.9, 'Red pain - Blue nopain - Green eff - Yellow mess')
plt.figtext(0.2, 0.03, 'Base: circle, Super: triangle, Normal: square')
# plt.show()
os.makedirs('./output/umap/lesion', exist_ok=True)
plt.savefig(f'./output/umap/lesion/{prj}_tsne.png')

d = {'name': ['base', 'super', 'super_mess', 'super_eff', 'normal'],
     'dist_value': [base_dist_value, super_dist_value, super_mess_dist_value, super_eff_dist_value, normal_dist_value],
     'dist_xy': [base_dist_xy, super_dist_xy, super_mess_dist_xy, super_eff_dist_xy, normal_dist_xy]}
df = pd.DataFrame(data=d)
# df.to_csv(f'./output/umap/lesion/{prj}.csv', index=False)