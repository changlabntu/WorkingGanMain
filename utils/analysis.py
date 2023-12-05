import os
import datetime
import numpy as np
import argparse
import tifffile as tiff
from skimage import morphology
from skimage.measure import euler_number, label
import surface_distance as surfdist
from tqdm import tqdm

def read_img(img_type, img_root):
    list = []
    if img_type == 'nii.gz':
        name = [_ for _ in os.listdir(img_root) if _.endswith('.nii.gz')]
        name.sort()
        for i in range(len(name)):
            img = sitk.ReadImage(os.path.join(img_root, name[i]))
            img = sitk.GetArrayFromImage(img)
            list.append(img[0, :, :])
    if img_type == 'tif':
        name = [_ for _ in os.listdir(img_root) if _.endswith('.tif')]
        try:
            name.sort(key=lambda x:int(x.split('_')[-1][:-4]))
        except:
            name.sort()
        for i in range(len(name)):
            # print(name[i])
            img = tiff.imread(os.path.join(img_root, name[i]))
            list.append(img)
    return list

def ROI_nptrd(list, trd, roll):
    x = np.array(list)
    if trd is not None:
        x[x >= trd] = 1
        x[x < trd] = 0
    if len(x) == 1 :
        x = x[0,:,:,:]
    if roll > 0:
        x = np.roll(x, roll, axis=2)
    print('Roll is ', roll)
    return x[z0:z1, y0:y1, x0:x1]

def dice(x, y):
    x = np.asarray(x).astype(np.bool_)
    y = np.asarray(y).astype(np.bool_)

    if x.shape != y.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

     # Compute Dice coefficient
    intersection = np.logical_and(x, y)

    dice_co = 2. * intersection.sum() / (x.sum() + y.sum())

    return dice_co

def centerline_3d(img):
    ske = morphology.skeletonize_3d(img)
    ske = ske.astype(np.uint8)
    return ske

def intersection(ske, img):
    result = ske * img
    sum = np.sum(ske)
    intersection = np.sum(result)
    return intersection/sum

def abs_error(list1, list2):
    if len(list1) == len(list2):
        diff = [abs(list1[i] - list2[i]) for i in range(len(list1))]
        total = sum(diff)
    return total, len(list1)

def euler(img, size):
    object_e8 = []
    hole_e8 = []
    for i in range(img.shape[0]):
        for y in range(0, int(img.shape[1]//size)):
            for x in range(0, int(img.shape[2]//size)):
                e8 = euler_number(img[i, y*size:y*size+size, x*size:x*size+size], connectivity=2)
                b0 = label(img[i, y*size:y*size+size, x*size:x*size+size], connectivity=2).max()
                b1 = b0 - e8
                object_e8.append(b0)
                hole_e8.append(b1)
    return object_e8, hole_e8


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help='Find train epoch or do all testing')
    args = parser.parse_args()
    betti_size = 128
    spacing = (1, 1, 1)
    (z0, z1, y0, y1, x0, x1) = (10, 600, 20, 2038, 20, 1010)  # crop ROI: (10, 600, 20, 2038, 20, 1010)

    # read img
    # ori:/media/usb-ST10000N_E000-3AP101_DD20221222012D-0:0/dayu/dataset/blood_vessel/segment/dataset/resizepseudo                  0.1275
    # ori_8x:/media/usb-ST10000N_E000-3AP101_DD20221222012D-0:0/dayu/dataset/blood_vessel/segment/training/label/train0yz8x.tif      0.1275
    # cyc:/media/usb-ST10000N_E000-3AP101_DD20221222012D-0:0/dayu/dataset/blood_vessel/segment/result/cyc4/13/result_img/pseudo      -0.75
    #img1_path = '/media/usb-ST10000N_E000-3AP101_DD20221222012D-0:0/dayu/dataset/blood_vessel/segment/dataset/resizepseudo'
    #img2_path = '/media/usb-ST10000N_E000-3AP101_DD20221222012D-0:0/dayu/dataset/blood_vessel/segment/result/cyc4/13/result_img/pseudo'
    img1_path = '/media/ghc/GHc_data2/N3Dreconstruction/cutF.tif'
    img2_path = '/media/ghc/GHc_data2/N3Dreconstruction/cutF2.tif'

    for trd_1 in [-0.75]:
        for trd_2 in [-0.8]:

            list1 = tiff.imread(img1_path)#read_img('tif', img1_path)
            # list2 = tiff.imread(img2_path)
            list2 = tiff.imread(img2_path)#read_img('tif', img2_path)
            print('Img1: ', img1_path, '\ntrd_1: ', trd_1)
            img1 = ROI_nptrd(list1, trd_1, 0)
            print('Img2: ', img2_path, '\ntrd_2: ', trd_2)
            img2 = ROI_nptrd(list2, trd_2, 0)

            # need parameter: img1, img2, betti_size, spacing
            if args.task == 'dice' or args.task == 'all' :
                start_time = datetime.datetime.now()


                dice_score = dice(img1, img2)
                print('The Dice is {}'.format(dice_score))

                end_time = datetime.datetime.now()
                during_time = end_time - start_time

            if args.task == 'cldice' or args.task == 'all':
                start_time = datetime.datetime.now()

                print('~~~ Doing clDice ~~~')
                print('ske1')
                ske1 = centerline_3d(img1)
                print('ske2')
                ske2 = centerline_3d(img2)
                precision = intersection(ske1, img2)
                recall = intersection(ske2, img1)

                clDice = (2 * precision * recall) / (precision + recall)
                print('The clDice is {}'.format(clDice))

                end_time = datetime.datetime.now()
                during_time = end_time - start_time
                print('Time: {}'.format(during_time))
            if args.task == 'euler' or args.task == 'all':
                start_time = datetime.datetime.now()

                print('~~~ Doing euler number ~~~')
                object_1_list, hole_1_list = euler(img1, betti_size)
                object_2_list, hole_2_list = euler(img2, betti_size)
                total_b0, len_b0 = abs_error(object_1_list, object_2_list)
                print('b0 sum: ', total_b0, ' & b0 len: ', len_b0)
                print('b0 error: ', total_b0 / len_b0)
                total_b1, len_b1 = abs_error(hole_1_list, hole_2_list)
                print('b1 sum: ', total_b1, ' & b1 len: ', len_b1)
                print('b1 error: ', total_b1 / len_b1)

                end_time = datetime.datetime.now()
                during_time = end_time - start_time
                print('Time: {}'.format(during_time))
            if args.task == 'distance' or args.task == 'all':
                start_time = datetime.datetime.now()

                print('~~~ Surface distance ~~~')
                img1_copy = img1.astype(np.bool_)
                img2_copy = img2.astype(np.bool_)
                surface_distances = surfdist.compute_surface_distances(img1_copy, img2_copy, spacing)
                hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
                print('Hausdorff_95_score = {}'.format(hd_dist_95))

                end_time = datetime.datetime.now()
                during_time = end_time - start_time
                print('Time: {}'.format(during_time))



# python analysis.py --task dice