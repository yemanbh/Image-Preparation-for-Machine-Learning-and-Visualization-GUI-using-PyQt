
import os
import xml.etree.ElementTree as ET
from skimage import io
import  numpy as np

# utility functions
from Utilits.Image_augmentation_util import get_rotation_maxtrix,rotate_image,rotate_landmark
from Utilits.xml_utilts import update_xml,get_type_center_from_XML
from Utilits.image_reading_utilits import read_dicom,read_image, read_nii


class AugmentImage:
    """
    A class definition for image augmentation
    """
    def __init__(self, data=None,
                 img_name=None,
                 appply_on_batch_files = False,
                 img_source_folder = None,
                 landmark_source_folder  = None,
                 landmark_path=None,
                 rotation_angle = 90,
                 dst_folder=None,
                 flip_vertical = False,
                 flip_horizental = False):

        """
        Class constructor

        :param data: input image
        :param img_name: input image name
        :param appply_on_batch_files: flag to apply to all files in folder or only to single file
        :param img_source_folder: input images source folder(for batch processing)
        :param landmark_source_folder:input landmarks source folder(for batch processing)
        :param landmark_path: direcvtory of landmark
        :param rotation_angle: roation angle for augmentation
        :param dst_folder: augmenteed images saving folder
        :param flip_vertical: Flag to flip vertically
        :param flip_horizental:Flag to flip horizontally
        """
        # image to be processed
        self.data = data
        self.image_name = img_name
        # land marks: row correspondece to one point
        self.landmark_path = landmark_path

        # apply to all image in a folder or to a single file
        self.appply_on_batch_files = appply_on_batch_files

        # images and xml source folders
        self.img_source_folder  = img_source_folder
        self.xml_source_folder = landmark_source_folder

        # dst folder
        self.dst_folder = dst_folder


        # rotation angle
        self.rotation_angle = rotation_angle

        # Fliping
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizental
        self.theta  = None
        #self.rotation_matrix  = None

        # create a folder to hold augmented images and xml
        if self.dst_folder is None or self.dst_folder is "" or self.dst_folder is []:
            self.dst_folder = os.getcwd()
        self.img_folder = os.path.join(self.dst_folder,'Augmented/Images')
        if os.path.exists(self.img_folder) is not True:
            os.makedirs(self.img_folder)
        self.xml_folder = os.path.join(self.dst_folder, 'Augmented/XML')
        if os.path.exists(self.xml_folder) is not True:
            os.makedirs(self.xml_folder)
        

    def apply_augmentation(self):
        """
        apply image augmentation
        :return: no return value
        """
        if self.appply_on_batch_files is False: # single image
            # get image extention
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]

            # apply augmentation
            self.do_augmentation()

        else: # apply to all images in the folder
            assert len(os.listdir(self.img_source_folder)) or \
                   len(os.listdir(self.xml_source_folder)) , 'source folder is empty'
            assert len(os.listdir(self.xml_source_folder)) == len(os.listdir(self.img_source_folder)),\
                'number of xml file and image files shoud be same, BUT, found different'

            img_files  = os.listdir(self.img_source_folder)
            xml_files = os.listdir(self.xml_source_folder)
            for images_name , xml_name in zip(img_files, xml_files):
                self.image_name = images_name
                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                # read an image
                name_ = os.path.join(self.img_source_folder,images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)

                # read XML file
                self.landmark_path  = os.path.join(self.xml_source_folder,xml_name)
                # apply augmentation
                self.do_augmentation()
        print('Augmentation done!')

    def do_augmentation(self):
        """
        apply requested augmentation and save file
        :return: no teturn vale
        """

        if self.flip_horizontal is True:
            angle = 180
            self.apply_rotation(angle, aug_type = str(angle))

        if self.flip_vertical is True:
            angle = 90
            self.apply_rotation(angle , aug_type = str(angle))
        if self.rotation_angle is not 0:
            self.apply_rotation(self.rotation_angle, aug_type = str(self.rotation_angle))

    def apply_rotation(self,angle , aug_type = 'rot'):
        """
        apply rotation on an image and landmark
        :param angle: ritation angle
        :param aug_type: type of augmeentation
        :return: none
        """
        # get rotation matrix
        rot_matrix, image_dim  = get_rotation_maxtrix(self.data, angle)
        # rotate an image
        rotated_image = rotate_image(self.data,rot_matrix,image_dim)

        # save rotated image
        image_path = os.path.join(self.img_folder,self.img_name_ + aug_type + '.png')
        
        #
        m = np.min(rotated_image)
        M = np.max(rotated_image - m)
        
        im_normalized = (rotated_image - m).astype('float32')/ float(M) * 255  # .astype()
        im_normalized = im_normalized.astype('uint8')
        io.imsave(image_path, im_normalized)

        if self.landmark_path is not None:
            # read the xml file
            xml_tree = ET.parse(self.landmark_path)
            # apply rotation to landmark points
            xml_tree_updated  = update_xml(xml_tree,rot_matrix)

            # save updated landmark
            xml_path = os.path.join(self.xml_folder, self.img_name_ +  aug_type +'.xml')
            xml_tree_updated.write(xml_path)




