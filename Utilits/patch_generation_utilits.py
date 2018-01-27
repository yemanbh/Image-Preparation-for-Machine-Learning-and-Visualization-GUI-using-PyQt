

import numpy as np

# patch extraction
from sklearn.feature_extraction.image import extract_patches



def extract_2d_patches(img_data, patch_shape, extraction_step, gt_data = None):
    """
    This code extracts 2D patches
    :param img_data: input image data
    :param patch_shape:  patch shape
    :param extraction_step:  patch extraction step or stride
    :param gt_data: ground truth data if any
    :return: generated patchs
    """

    # empty matrix to hold patches
    imgs_patches_ = np.empty(shape=[0, patch_shape[0], patch_shape[1]], dtype='int16')
    gt_patches_ = np.empty(shape=[0, patch_shape[0], patch_shape[1]], dtype='int16')

    img_patches = extract_patches(img_data, patch_shape, extraction_step)

    if gt_data is not None:

        gt_patches = extract_patches(gt_data, patch_shape, extraction_step)

        # Select nonzero patches
        Sum = np.sum(gt_patches, axis=(2, 3))
        rows, cols = np.nonzero(Sum)
        # number of n0m zero patches
        N = len(rows)
        # select nonzeropatches index
        if N is not 0:
            selcted_img_patches = img_patches[rows, cols, :, :]
            selcted_gt_patches = gt_patches[rows, cols, :, :]
    
            # update database
            imgs_patches_ = np.append(imgs_patches_, selcted_img_patches, axis=0)
            gt_patches_ = np.append(gt_patches_, selcted_gt_patches, axis=0)
        return imgs_patches_, gt_patches_

    else:
        # Select nonzero patches
        Sum = np.sum(img_patches, axis=(2, 3))
        rows, cols = np.nonzero(Sum)
        # select nonzeropatches index
        N  = len(rows)
        if N is not 0:
            selcted_img_patches = img_patches[rows, cols, :, :]
            # update database
            imgs_patches_ = np.append(imgs_patches_, selcted_img_patches, axis=0)

        return imgs_patches_

def extract_3d_patches(img_data, patch_shape, extraction_step, gt_data  = None):
    """
    This code extracts 3D patches
    :param img_data: input image data
    :param patch_shape:  patch shape
    :param extraction_step:  patch extraction step or stride
    :param gt_data: ground truth data if any
    :return: generated patchs
    """
    # empty matrix to hold patches
    imgs_patches = np.empty(shape=[0, patch_shape[0], patch_shape[1], patch_shape[2]], dtype='int16')
    gt_patches_per_volume = np.empty(shape=[0, patch_shape[0], patch_shape[1], patch_shape[2]], dtype='int16')

    img_patches = extract_patches(img_data, patch_shape, extraction_step)

    if gt_data is not None :
        gt_patches = extract_patches(gt_data, patch_shape, extraction_step)
        # Select nonzero patches
        Sum = np.sum(gt_patches, axis=(3, 4, 5))
        rows, cols, depths = np.nonzero(Sum)
        # number of n0m zero patches
        N = len(rows)
        # select nonzeropatches index
        #    rand_index = random.sample(range(1,N), 200)
        #    rows = rows[rand_index]
        #    cols = cols[rand_index]
        #    depths = depths[rand_index]
        if N is not 0:
            selcted_img_patches = img_patches[rows, cols, depths, :, :, :]
            selcted_gt_patches = gt_patches[rows, cols, depths, :, :, :]
    
            # update database
            imgs_patches = np.append(imgs_patches, selcted_img_patches, axis=0)
            gt_patches = np.append(gt_patches_per_volume, selcted_gt_patches, axis=0)
        return imgs_patches, gt_patches
    else:

        Sum = np.sum(img_patches, axis=(3, 4, 5))
        rows, cols, depths = np.nonzero(Sum)
        # number of n0m zero patches
        N = len(rows)
        if N is not 0:
            selcted_img_patches = img_patches[rows, cols, depths, :, :, :]
            # update database
            imgs_patches = np.append(imgs_patches, selcted_img_patches, axis=0)

        return imgs_patches
