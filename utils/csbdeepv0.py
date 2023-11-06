import numpy as np
import glob, os, time
import tifffile as tiff
from PIL import Image
import scipy.ndimage


def to_8bit(x):
    x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


def torch_upsample(x, size, normalize):
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    x = x.permute(0, 1, 3, 4, 2)
    up = nn.Upsample(size=(size[0], size[1], x.shape[4]), mode='trilinear', align_corners=True)
    x = up(x)
    if normalize:
        x = transforms.Normalize(0.5, 0.5)(x)
    x = x.permute(0, 1, 4, 2, 3).squeeze().numpy()
    x = x.astype(np.float16)
    return x


def quick_compare(x0, y0, shade=0.5):
    x = x0 - x0.min()
    x = x / x.max()
    y = y0 - y0.min()
    y = y / y.max()
    combine = np.concatenate([np.expand_dims(x, 2)] * 3, 2)
    combine[:, :, 0] = np.multiply(x, 1 - shade*y)
    combine[:, :, 2] = np.multiply(x, 1 - shade*y)
    return combine


def pil_upsample(x, size):
    y = []
    for s in range(x.shape[0]):
        temp = Image.fromarray(x[s, ::])
        temp = temp.resize(size=(size[1], size[0]))
        y.append(np.expand_dims(np.array(temp), 0))
    y = np.concatenate(y, 0)
    return y


def run_csbdeep(tif, deconv, osize, rates):
    import tensorflow as tf
    model = tf.saved_model.load('CSBDeep/param')

    for rate in rates:
        print(osize)
        size = [int((rate * s) // 4 * 4) for s in osize]
        print(size)

        npy = pil_upsample(tif, size=size)

        npy = npy / npy.max()
        all = []
        tini = time.time()

        for s in range(npy.shape[0]):
            print(s)
            fiber = npy[s, ::]  # (H, W)
            fiber = np.expand_dims(fiber, 0)  # [RGB_channel=1, h, w]
            fiber = np.expand_dims(fiber, 3)  # [batch=1, RGB_channel=1, h, w]
            #fiber = np.transpose(fiber, (0, 2, 3, 1)) # [batch, h, w, RGB_channel=1]
            fiber = tf.constant(fiber, dtype=tf.float32)

            # print(list(model.signatures))
            result = model.signatures['serving_default'](fiber)

            model.signatures['serving_default'](tf.constant(fiber, dtype=tf.float32))

            # result have 2 channel, we want channel 0
            #result_to1 = (np.array(result['output'][:, :, :, 0]) - np.array(result['output'][:,:,:,0]).min()) / np.array(result['output'][:,:,:,0]).max()
            out = np.array(result['output'][:, :, :, :])
            all.append(out)

        print(time.time() - tini)

        all = np.concatenate(all, 0)
        tiff.imsave(deconv + str(int(rate*100)).zfill(3) + '.tif', all[:, :, :, 0].astype(np.float16))


def get_fusion(img_path, deconv, osize, destination):
    import torch
    import torch.nn as nn
    """
    Here is the function to resemble the mask you want
    """
    deconv = sorted(glob.glob(deconv + '*'))[:]
    allx = []

    # read all the deconv tifs and then concatenate them together
    for t in deconv[::]:
        print(t)
        x = tiff.imread(t)
        x = torch_upsample(x.astype(np.float32), size=osize, normalize=True)
        x = x - x.min()
        x = x / x.max()
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)
        allx.append(np.expand_dims(x, 3))
    allx = np.concatenate(allx, 3)

    allx = torch.from_numpy(allx).type(torch.FloatTensor)

    # fft
    xfft = torch.fft.fft(allx[:, :, :, :], dim=3)
    xfft = torch.real(xfft)

    tiff.imsave('xfftrori.tif', xfft[:,:,:,0].numpy())

    # voting = (all the deconv >= threshold).sum()
    #weak = (allx[:, :, :, :] >= 0.07).sum(3)
    #tiff.imsave(root + 'pseudo.tif', weak.astype(np.uint8))



def get_fft_again(x):
    root = '/home/ubuntu/Data/Dataset/paired_images/Fly0B/'
    l = ['A', 'B', 'C', 'D']
    tifs = [tiff.imread(root + x + '.tif') for x in l]


if __name__ == '__main__':
    if 0:
        #root = '/media/ghc/GHc_data2/BRC/40xdescar/'
        #img_path = root + 'original_cropped.tif'
        root = '/home/ubuntu/Data/Dataset/paired_images/40xhan/'
        img_path = root + 'xyori05.tif'
    elif 0:
        root = '/media/ghc/GHc_data2/BRC/4xfly/'
        img_path = root + 'roiA.tif'
    elif 1:
        #root = '/media/ExtHDD01/Dataset/paired_images/organoidA/'
        root = '/mnt/nas/Data/Data_GHC/BRC/organoid/organoidA/'
        img_path = root + 'xyori.tif'
    elif 0:
        root = '/home/ubuntu/Data/Dataset/BRC/Henry/221215/roi/'
        img_path = root + 'x.tif'

    deconv = root + 'deconv/'

    tif = tiff.imread(img_path)[:, :, :]#[500:510, :1024, 1024:]
    osize = tif.shape[1:3]
    print(osize)

    # run CSSBDeep
    os.makedirs(deconv, exist_ok=True)

    # thresholding
    #tif[tif >= 1000] = 1000

    # downsampling rates into CSBDeep
    #rates = [x / 100 for x in list(range(5, 30, 2))] + [x / 100 for x in list(range(30, 301, 10))]

    #rates = [x / 100 for x in list(range(5, 30, 2))] + [x / 100 for x in list(range(30, 101, 10))]

    #rates = [5**(x/10) / 100 for x in list(range(10, 30, 1))]

    #rates = [0.5]#[x / 100 for x in list(range(10, 101, 5))][:]
    rates = [x / 100 for x in list(range(10, 101, 10))]

    # run csb deep
    #run_csbdeep(tif, deconv, osize, rates)
    get_fusion(img_path, deconv, osize, destination=root)
