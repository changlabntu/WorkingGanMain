import torch.utils.data as data
import albumentations as A
from torchvision import transforms
from PIL import Image
import cv2
import os
import glob
import torch
import numpy as np
import tifffile as tiff
import random
import torch.nn as nn

# from model.unet import Unet

# from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


def to_8bit(img):
    img = np.array(img)
    img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)  # images data root
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0, 2) < 1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class PainDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader):
        imgs = sorted(glob.glob(data_root))  # images data root

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs[:]
        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
            # A.ToTensor()
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

        self.model = torch.load('submodels/80.pth', map_location='cpu').eval()
        self.kernal_size = 13
        self.conv = nn.Conv2d(1,1,self.kernal_size,1,self.kernal_size//2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        self.kernal_size_2 = 13
        self.conv_2 = nn.Conv2d(1,1,self.kernal_size_2,1,self.kernal_size_2//2)
        self.conv_2.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size_2, self.kernal_size_2))
        self.conv_2.bias = nn.Parameter(torch.Tensor([0]))

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5

        transformed = self.tfs(image=img)
        img = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)
        mask, masked_inside, masked_outside, pred = self.transform_mask(tiff.imread(path.replace('bp', 'apeff/apeff').replace('/ap/', '/apeff/apeff/')),
                                                                        tiff.imread(path.replace('bp', 'apmean').replace('/ap/', '/apmean/').replace('.', '_100.0.')),
                                                                        img
                                                                        )
        mask = torch.unsqueeze(mask, 0)
        # mask = torch.unsqueeze(self.transform_mask(tiff.imread(path.replace('bp', 'meanA/meanA'))), 0)
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask
        cond_image = torch.cat([cond_image, masked_inside.squeeze(0), masked_outside.squeeze(0), pred.squeeze(0)], 0) # (4, 384, 384)
        # print(cond_image.shape)
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, mask, mask_2=None, img=None):
        threshold = 0

        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)
        if img is not None:
            img = torch.unsqueeze(img, 0)
            pred = self.model(img)
            pred = torch.argmax(pred, 1, True)
            one_pred = (pred > 0).type(torch.uint8)

        if mask_2 is not None:
            threshold = 0.07 * self.kernal_size_2 * self.kernal_size_2
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            mask += mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

            if img is not None:
                tmp = mask.type(torch.float32) - one_pred.type(torch.float32)
                masked_outside = (tmp > 0).type(torch.uint8) * torch.randn_like(img) + (1 - one_pred) * img
                masked_inside = one_pred * img + one_pred * (mask_2 > 0).type(torch.uint8) * torch.randn_like(img)
            # tiff.imwrite('img.tif', to_8bit(img))
            # tiff.imwrite('pred.tif', to_8bit(pred))
            # tiff.imwrite('see.tif', to_8bit(masked_inside))
            # tiff.imwrite('seeee.tif', to_8bit(masked_outside))
        if img is not None:
            return mask, masked_inside, masked_outside, pred
        else:
            return mask


class PainValidationDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader):
        all = sorted(glob.glob(data_root))
        ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
        # ids = [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528]
        # just for more sample to test
        # rand_ids = [random.randint(0, 29999) for y in range(40)]
        # ids += rand_ids
        imgs = [all[i] for i in ids]

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
            # A.ToTensor()
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5

        transformed = self.tfs(image=img)
        img = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)
        mask = torch.unsqueeze(
            self.transform_mask(tiff.imread(path.replace('ap', 'apeff/apeff')),
                                tiff.imread(path.replace('ap', 'apmean').replace('.', '_100.0.'))), 0)
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def transform_mask(self, mask, mask_2=None):
        threshold = 0

        # mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)
        if mask_2 is not None:
            threshold = 0.10
            mask_2 = cv2.GaussianBlur(mask_2, (3, 3), 0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask = np.array(mask) + mask_2
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)
        return mask


class dagm4(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader,
                 selected_attrs=['cls_type0', 'cls_type1'], mode='train'):
        self.data_path = '/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data'
        att_list = \
        open('/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data/DAGM4.txt', 'r', encoding='utf-8').readlines()[
            1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt('/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data/DAGM4.txt', skiprows=2, usecols=[0],
                            dtype=np.str)
        labels = np.loadtxt('/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data/DAGM4.txt', skiprows=2,
                            usecols=atts, dtype=np.int)
        self.mode = mode
        tmp_index = [num for num, x in enumerate(labels) if x[1] == 1]

        labels = np.concatenate([np.expand_dims(labels[x], 0) for x in tmp_index], 0)
        images = [images[x] for x in tmp_index]  # 兩個都是150個

        if self.mode == 'train':
            self.images = images[:140]
            self.labels = labels[:140]

            self.tf = A.Compose([
                A.RandomResizedCrop(height=256, width=256, scale=(1, 1.2)),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if self.mode == 'valid':
            self.images = images[800:1050]
            self.labels = labels[800:1050]

            self.tf = A.Compose([
                A.RandomResizedCrop(height=256, width=256, scale=(1, 1.2)),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        if self.mode == 'test':
            self.images = images[1050:]
            self.labels = labels[1050:]
            self.tf = A.Compose([
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.length = len(self.images)

    def __getitem__(self, index):
        ret = {}
        # print(os.listdir(os.path.join(self.data_path, 'DAGM')))
        img = np.expand_dims(np.array(
            Image.open(os.path.join(self.data_path, self.images[index].replace('\\', '/').replace('png', 'PNG')))), 2)
        img = np.concatenate([img, img, img], 2)
        mask = np.array(Image.open(
            os.path.join(self.data_path, 'mask_' + self.images[index].replace('\\', '/').replace('png', 'PNG'))))[:, :,
               0]

        att = torch.tensor((self.labels[index] + 1) // 2)
        bbox_f = open(
            os.path.join(self.data_path, 'bbox_' + self.images[index].replace('\\', '/').split('.')[0] + '.txt'), 'r')
        bbox = bbox_f.read()

        try:
            import random
            x_mid, y_mid, x_w, y_h = bbox.split(' ')[1:]
            y_h = y_h.split('\\')[0]
            x_l = int((float(x_mid) - (float(x_w) / 2)) * img.shape[0])
            x_r = int((float(x_mid) + (float(x_w) / 2)) * img.shape[0])
            y_u = int((float(y_mid) - (float(y_h) / 2)) * img.shape[1])
            y_d = int((float(y_mid) + (float(y_h) / 2)) * img.shape[1])
            if x_r <= 256 and y_d <= 256:
                random_x = random.randint(0, x_l)
                random_y = random.randint(0, y_u)
                img = img[random_y:random_y + 256, random_x:random_x + 256]
                mask = mask[random_y:random_y + 256, random_x:random_x + 256]
            elif x_r >= 256 and y_d <= 256:
                random_x = random.randint(x_r, img.shape[0])
                random_y = random.randint(0, y_u)
                img = img[random_y:random_y + 256, random_x - 256:random_x]
                mask = mask[random_y:random_y + 256, random_x - 256:random_x]
            elif x_r <= 256 and y_d >= 256:
                random_x = random.randint(0, x_l)
                random_y = random.randint(y_d, img.shape[1])
                img = img[random_y - 256:random_y, random_x:random_x + 256]
                mask = mask[random_y - 256:random_y, random_x:random_x + 256]
            elif x_r >= 256 and y_d >= 256:
                random_x = random.randint(x_r, img.shape[0])
                random_y = random.randint(y_d, img.shape[1])
                img = img[random_y - 256:random_y, random_x - 256:random_x]
                mask = mask[random_y - 256:random_y, random_x - 256:random_x]
        except:
            random_x = random.randint(0, img.shape[0] / 2)
            random_y = random.randint(0, img.shape[1] / 2)
            img = img[random_x:random_x + 256, random_y:random_y + 256]
            mask = mask[:256, :256]
        transformed = self.tf(image=img, mask=mask)

        img = torch.Tensor(np.transpose(transformed['image'], (2, 0, 1)))[0:1, ::]
        mask = torch.IntTensor(transformed['mask'] / 4).unsqueeze(0)

        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = self.images[index].split("\\")[1]
        return ret

    def __len__(self):
        return self.length


class dagm6(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader,
                 selected_attrs=['cls_type0', 'cls_type1'], mode='train'):
        self.data_path = '/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data'
        att_list = \
        open('/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data/DAGM6.txt', 'r', encoding='utf-8').readlines()[
            1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt('/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data/DAGM6.txt', skiprows=2, usecols=[0],
                            dtype=np.str)
        labels = np.loadtxt('/home/ziyi/Projects/AttGAN-Pytorch/AttGAN-PyTorch/data/DAGM6.txt', skiprows=2,
                            usecols=atts, dtype=np.int)
        self.mode = mode
        tmp_index = [num for num, x in enumerate(labels) if x[1] == 1]

        labels = np.concatenate([np.expand_dims(labels[x], 0) for x in tmp_index], 0)
        images = [images[x] for x in tmp_index]  # 兩個都是150個

        if self.mode == 'train':
            self.images = images[:140]
            self.labels = labels[:140]

            self.tf = A.Compose([
                A.RandomResizedCrop(height=256, width=256, scale=(1, 1.2)),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if self.mode == 'valid':
            self.images = images[800:1050]
            self.labels = labels[800:1050]

            self.tf = A.Compose([
                A.RandomResizedCrop(height=256, width=256, scale=(1, 1.2)),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        if self.mode == 'test':
            self.images = images[140:]
            self.labels = labels[140:]
            self.tf = A.Compose([
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.length = len(self.images)

    def __getitem__(self, index):
        ret = {}
        # print(os.listdir(os.path.join(self.data_path, 'DAGM')))
        img = np.expand_dims(np.array(
            Image.open(os.path.join(self.data_path, self.images[index].replace('\\', '/').replace('png', 'PNG')))), 2)
        img = np.concatenate([img, img, img], 2)
        mask = np.array(Image.open(
            os.path.join(self.data_path, 'mask_' + self.images[index].replace('\\', '/').replace('png', 'PNG'))))[:, :,
               0]

        att = torch.tensor((self.labels[index] + 1) // 2)
        bbox_f = open(
            os.path.join(self.data_path, 'bbox_' + self.images[index].replace('\\', '/').split('.')[0] + '.txt'), 'r')
        bbox = bbox_f.read()

        try:
            import random
            x_mid, y_mid, x_w, y_h = bbox.split(' ')[1:]
            y_h = y_h.split('\\')[0]
            x_l = int((float(x_mid) - (float(x_w) / 2)) * img.shape[0])
            x_r = int((float(x_mid) + (float(x_w) / 2)) * img.shape[0])
            y_u = int((float(y_mid) - (float(y_h) / 2)) * img.shape[1])
            y_d = int((float(y_mid) + (float(y_h) / 2)) * img.shape[1])
            if x_r <= 256 and y_d <= 256:
                random_x = random.randint(0, x_l)
                random_y = random.randint(0, y_u)
                img = img[random_y:random_y + 256, random_x:random_x + 256]
                mask = mask[random_y:random_y + 256, random_x:random_x + 256]
            elif x_r >= 256 and y_d <= 256:
                random_x = random.randint(x_r, img.shape[0])
                random_y = random.randint(0, y_u)
                img = img[random_y:random_y + 256, random_x - 256:random_x]
                mask = mask[random_y:random_y + 256, random_x - 256:random_x]
            elif x_r <= 256 and y_d >= 256:
                random_x = random.randint(0, x_l)
                random_y = random.randint(y_d, img.shape[1])
                img = img[random_y - 256:random_y, random_x:random_x + 256]
                mask = mask[random_y - 256:random_y, random_x:random_x + 256]
            elif x_r >= 256 and y_d >= 256:
                random_x = random.randint(x_r, img.shape[0])
                random_y = random.randint(y_d, img.shape[1])
                img = img[random_y - 256:random_y, random_x - 256:random_x]
                mask = mask[random_y - 256:random_y, random_x - 256:random_x]
        except:
            random_x = random.randint(0, img.shape[0] / 2)
            random_y = random.randint(0, img.shape[1] / 2)
            img = img[random_x:random_x + 256, random_y:random_y + 256]
            mask = mask[:256, :256]
        transformed = self.tf(image=img, mask=mask)

        img = torch.Tensor(np.transpose(transformed['image'], (2, 0, 1)))[0:1, ::]
        mask = torch.IntTensor(transformed['mask'] / 4).unsqueeze(0)

        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        try:
            ret['path'] = self.images[index].split("\\")[1]
        except:
            ret['path'] = self.images[index].split("/")[1]
        return ret

    def __len__(self):
        return self.length


class FlyDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = glob.glob(data_root)  # images data root

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
            # A.ToTensor()
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5

        transformed = self.tfs(image=img)
        img = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)
        # inside mask is 1
        mask = torch.unsqueeze(
            self.sample_mask(), 0).float()

        # mask = torch.unsqueeze(self.transform_mask(tiff.imread(path.replace('bp', 'meanA/meanA'))), 0)
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def sample_mask(self, shape=[256, 256]):
        mask = np.zeros((shape[0], shape[1]))
        cut = random.randint(64, 128)
        if random.random() < 0.5:
            mask[cut: cut + int(shape[0] / 4), :] = True
        else:
            mask[:, cut:cut + int(shape[1] / 4)] = True

        return torch.from_numpy(mask)


if __name__ == "__main__":
    train_dataset = PainDataset("/media/ziyi/Dataset/OAI_pain/full/bp/*", mask_config={"mask_mode": "hybrid"})

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True)
    for i in train_dataloader:
        # print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        # print(i["cond_image"].shape)
        # print(i["mask_image"].shape)
        assert 0
