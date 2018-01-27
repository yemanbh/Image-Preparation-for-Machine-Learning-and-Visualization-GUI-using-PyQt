import dicom
# for nii file
import nibabel as nib
import numpy as np
from skimage import  io

def read_dicom(dicom_path):
    """
    read dicom image
    :param dicom_path: dicom image dir
    :return: dicom image as numpy array
    """
    RefDs = dicom.read_file(dicom_path)
    dicom_im = RefDs.pixel_array
    #plt.imshow(X, cmap=plt.cm.gray)
    return  dicom_im

def read_nii(nii_path):
    """
    read nii file
    :param nii_path: image path
    :return: image as numpy array
    """
    img = nib.load(nii_path)
    img_image_data = img.get_data()
    image_data = np.squeeze(img_image_data, axis=3)

    return image_data

def read_image(image_path):
    """
    reads other supported imnage formats: .PNG,.png, .tif,
    :param image_path:image dir
    :return: image as numpy array
    """
    return io.imread(image_path)


