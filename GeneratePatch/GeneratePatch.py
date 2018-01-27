
from __future__ import print_function
import os
import math

from skimage import io

#import matplotlib.pyplot as plt
import numpy as np
import random


from sklearn.metrics.pairwise import euclidean_distances

from Utilits.image_reading_utilits import read_dicom,read_image, read_nii
from Utilits.xml_utilts import get_type_center_from_XML
from Utilits.patch_generation_utilits import extract_2d_patches,extract_3d_patches

class GeneratePatch:
    """This is a class to generate pathces from an image. It it has methods to extract 2D, 2.5D and 3D patches
    using landmark as a background, image as a background and withoutr baack ground

     """
    def __init__(self, data=None,
                 name = None,
                 landmark = None,

                 appply_on_batch_files=False,
                 img_source_folder=None,
                 landmark_source_folder=None,
                 landmark_path=None,

                 patch_type = '2D', 
                 patch_dim = (32,32,0),
                 patch_percntg = 100, 
                 num_tumor_patch = None,
                 num_nontumor_patch = None,
                 save_file_type='.npy',
                 dst_folder = None,

                 gt_path_source_folder = None,
                 gt_img_path = None,
                 stride = None,
                 ground_truth_type=None):



        # initialization
        # patch extraction step
        self.stride = stride

        # apply to single file or to files in folder
        self.appply_on_batch_files  = appply_on_batch_files

        self.landmark_path = landmark_path
        self.gt_img_path = gt_img_path

        # images and xml source folders
        self.img_source_folder  = img_source_folder
        self.xml_source_folder = landmark_source_folder
        self.gt_path_source_folder = gt_path_source_folder

        # type of ground truth
        self.ground_truth_type = ground_truth_type

        
        # image to be processed
        self.data = data
        
        self.image_name = name

        #land marks: row correspondece to one point
        self.landmark = landmark

        # patch dimension
        self.patch_dim = patch_dim

        # patch type
        self. patch_type = patch_type

        # patch tumor and non tumor percentage, tumor/ non tumor
        self.patch_percntg =  patch_percntg
        self.num_tumor_patch  = num_tumor_patch
        self.num_nontumor_patch  = num_nontumor_patch
        
        # patches
        self.dst_folder  = dst_folder
        
        # image saave file type
        self.save_file_type = save_file_type

        # folder for saving patches generated
        if self.dst_folder is None or self.dst_folder is "" or self.dst_folder is []:
            self.dst_folder = os.getcwd()

        self.generated_patches_folder = os.path.join(self.dst_folder,'Patches_generated')
        if os.path.exists(self.generated_patches_folder) is False:
            os.mkdir(self.generated_patches_folder)
    
    def patch_generator_main(self):

        """ This is  the function called from GUI
                        and
        depending on the dimension and type of patch requested it will execuete corresponding class method.
        """

        # check the type of patch

        if self. patch_type=='2D': # for 2D patch extraction

            # check patch dimension
            assert self.patch_dim[2] is 0,'patch type and dimention conflict, patch depth should be 0'
            if self.ground_truth_type is None:
                self.extractPatch_2D_patchs_no_gt()

            # check ground truth type

            elif self.ground_truth_type is 'gt_mask':
                self.extractPatch_2D_patchs_mask_gt()

            elif self.ground_truth_type is 'landmark':
                self.extractPatch_2D_patchs_landmark()

        elif self.patch_type=='2.5D':# for 2.5D patch extraction
            # check ground truth type
            if self.ground_truth_type is None:
                self.extractPatch_2D5_patchs_no_gt()
            elif self.ground_truth_type is 'gt_mask':
                self.extractPatch_2D5_patchs_mask_gt()

        elif self.patch_type == '3D' and self.patch_dim[2] is not 0: # for 3D patch extraction
            # check ground truth type
            if self.ground_truth_type is None:
                self.extractPatch_3D_patchs_no_gt()
            elif self.ground_truth_type is 'gt_mask':
                self.extractPatch_3D_patchs_mask_gt()


    def extractPatch_2D_patchs_no_gt(self):

        if self.appply_on_batch_files is False:  # single image

            # get image name and extension
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]

            # check dimenation of the data
            Dim  = self.data.shape
            self.data  = self.to_8bit_(self.data)

            # 2D image
            if len(Dim) is 2:
                # extract patches
                im_patches = extract_2d_patches(self.data , self.patch_dim[:2],self.stride, gt_data=None)

                # patch name
                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                #save as npy file
                np.save(path_, im_patches)
            else: # image is 3D and apply per slice
                P = np.empty(shape=[0, self.patch_dim[0], self.patch_dim[1]], dtype='int16')
                for i in range(Dim[2]):
                    # extract slices per slice
                    data_ = self.to_8bit_(self.data[:,:,i])
                    im_patches = extract_2d_patches(data_, self.patch_dim[:2], self.stride, gt_data=None)
                    P = np.append(P, im_patches, axis=0)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                np.save(path_, P)



        else: # for batch of files
            # check if folder contains file
            assert len(os.listdir(self.img_source_folder)), 'source folder is empty'

            # read all  files
            img_files = os.listdir(self.img_source_folder)
            for images_name in img_files:
                self.image_name = images_name

                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                name_ = os.path.join(self.img_source_folder, images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)


                # check dimenation of the data
                Dim = self.data.shape

                # 2D image
                if len(Dim) is 2:
                    im_patches = extract_2d_patches(self.data, self.patch_dim[:2], self.stride, gt_data=None)

                    path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                    np.save(path_, im_patches)
                else:  # image is 3D and apply per slice
                    P = np.empty(shape=[0, self.patch_dim[0], self.patch_dim[1]], dtype='int16')
                    for i in range(Dim[2]):
                        # extract slices per slice
                        im_patches = extract_2d_patches(self.data[:, :, i], self.patch_dim[:2], self.stride,
                                                        gt_data=None)
                        P = np.append(P, im_patches, axis=0)

                    path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                    np.save(path_, P)


    def extractPatch_2D_patchs_mask_gt(self):
        if self.appply_on_batch_files is False:  # single image
            # get image extension
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]

            # for gt image
            gt_image_name = self.gt_img_path
            # get image extension
            gt_image_name = os.path.split(gt_image_name)
            gt_image_name = gt_image_name[-1]
            gt_img_name_split = gt_image_name.split(".")
            # gt_image_name = img_name_split[0]
            gt_img_extension = gt_img_name_split[-1]

            # Read an image
            name_ = os.path.join(self.gt_img_path)
            if gt_img_extension == 'dcm':
                self.gt_data = read_dicom(name_)
            elif gt_img_extension == 'nii' or gt_img_extension == 'gz' or gt_img_extension == 'nii.gz':
                self.gt_data = read_nii(name_)
            else:
                self.gt_data = read_image(name_)

            Dim  = self.data.shape
            if len(Dim) is 2: # 2D image
                # extract patches
                im_patches, gt_patches = extract_2d_patches(self.data , self.patch_dim[:2],self.stride, gt_data=self.gt_data)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                np.save(path_, im_patches)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

                np.save(path_, gt_patches)

            else: # image is 3D and apply per slice
                P_im = np.empty(shape=[0, self.patch_dim[0], self.patch_dim[1]], dtype='int16')
                P_gt = np.empty(shape=[0, self.patch_dim[0], self.patch_dim[1]], dtype='int16')
                for i in range(Dim[2]):
                    # extract slices per slice
                    im_patches, gt_patches = extract_2d_patches( self.data[:,:,i], self.patch_dim[:2], self.stride,  gt_data = self.gt_data[:,:,i])
                    P_im = np.append(P_im, im_patches, axis=0)
                    P_gt = np.append(P_gt, gt_patches, axis=0)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                np.save(path_, P_im)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

                np.save(path_, P_gt)

        else:  # for batch of files
            assert len(os.listdir(self.img_source_folder)) or \
                   len(os.listdir(self.gt_path_source_folder)), 'source folder is empty'
            assert len(os.listdir(self.gt_path_source_folder)) == len(os.listdir(self.img_source_folder)), \
                'number of ground truth file and image files shoud be same, BUT, found different!'

            img_files = os.listdir(self.img_source_folder)
            gt_files = os.listdir(self.gt_path_source_folder)
            for images_name, gt_name in zip(img_files, gt_files):
                self.image_name = images_name
                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                name_ = os.path.join(self.img_source_folder, images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)

                # Convert to 8 bit
                self.data = self.to_8bit_(self.data)

                # for gt image
                gt_image_name = gt_name
                # get image extension
                gt_image_name = os.path.split(gt_image_name)
                gt_image_name = gt_image_name[-1]
                gt_img_name_split = gt_image_name.split(".")
                gt_image_name = gt_img_name_split[0]
                gt_img_extension = gt_img_name_split[-1]

                name_ = os.path.join(self.gt_path_source_folder, gt_name)
                if gt_img_extension == 'dcm':
                    self.gt_data = read_dicom(name_)
                elif gt_img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.gt_data = read_nii(name_)
                else:
                    self.gt_data = read_image(name_)

                Dim = self.data.shape

                # 2D image
                if len(Dim) is 2:
                    im_patches, gt_patches = extract_2d_patches(self.data, self.patch_dim[:2], self.stride,
                                                                gt_data=self.gt_data)

                    path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                    np.save(path_, im_patches)

                    path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

                    np.save(path_, gt_patches)

                else:  # image is 3D and apply per slice
                    P_im = np.empty(shape=[0, self.patch_dim[0], self.patch_dim[1]], dtype='int16')
                    P_gt = np.empty(shape=[0, self.patch_dim[0], self.patch_dim[1]], dtype='int16')
                    for i in range(Dim[2]):
                        # extract slices per slice
                        im_patches, gt_patches = extract_2d_patches(self.data[:,:,i], self.patch_dim[:2], self.stride,
                                                                    gt_data=self.gt_data[:,:,i])
                        P_im = np.append(P_im, im_patches, axis=0)
                        P_gt = np.append(P_gt, gt_patches, axis=0)

                    path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                    np.save(path_, P_im)

                    path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

                    np.save(path_, P_gt)

        print('Patch Generation done!')


    def extractPatch_2D_patchs_landmark(self):
        """ This function extraccts 2D patches using land marks from an XML file"""

        if self.appply_on_batch_files is False: # single image
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]

            # convert image to 8 bit
            self.data = self.to_8bit_(self.data)

            self.do_2D_patch_extraction()

        else:
            assert len(os.listdir(self.img_source_folder)) or \
                   len(os.listdir(self.xml_source_folder)), 'source folder is empty'
            assert len(os.listdir(self.xml_source_folder)) == len(os.listdir(self.img_source_folder)), \
                'number of xml file and image files shoud be same, BUT, found different'

            img_files = os.listdir(self.img_source_folder)
            xml_files = os.listdir(self.xml_source_folder)
            for images_name, xml_name in zip(img_files, xml_files):
                self.image_name = images_name
                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                name_ = os.path.join(self.img_source_folder, images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)

                # convert image to 8 bit
                self.data = self.to_8bit_(self.data)

                # read XML file
                self.landmark_path = os.path.join(self.xml_source_folder, xml_name)
                lmark = get_type_center_from_XML(self.landmark_path)

                self.landmark = lmark

                # apply augmentation
                self.do_2D_patch_extraction()
        print('Patch Generation done!')


    def do_2D_patch_extraction(self):

        # prepare a folder to store patches
        path_ = os.path.join(self.generated_patches_folder, self.img_name_)

        if os.path.exists(path_) is not True:
            os.mkdir(path_)

        # record all land mark points center, later I will use them to compute center of non landmark points
        landmarks_center = np.empty((0, 2), 'uint16')

        # for all of patch catagories
        for patch_type, land_mark in self.landmark.items():

            # record land mark
            landmarks_center = np.append(landmarks_center, land_mark, axis=0)
            # create a folder for each type if it didn't exist
            patch_path = os.path.join(path_, patch_type)
            if os.path.exists(patch_path) is not True:
                os.mkdir(patch_path)

            if land_mark.shape[1] == 2:  # if the image is 2D

                # Number of patches to be extracted for the given patch category
                # it is the minimum of numner of center points given in the xml fil and number of tumor patches specified
                # in the GUI
                N = min(land_mark.shape[0], self.num_tumor_patch)

                # create empty matrix for npy file
                if self.save_file_type == '.npy':
                    patches_extracted = np.empty((self.patch_dim[0], self.patch_dim[1], 0), 'float32')

                # itrate over all landmarks
                for i in range(N):

                    # center point of patch

                    
                    C= int(land_mark[i, 0])  # row
                    R = int(land_mark[i, 1])  # column

                    # check if patch can be fix with this these center point and make adjustment
                    check_coordinate = self.check_coordinate_patch(R, C)

                    # get patch
                    patch_im = self.data[check_coordinate[0]: check_coordinate[1],
                               check_coordinate[2]:check_coordinate[3]]

                    # check file save type
                    if self.save_file_type == '.png':
                        # patch name
                        p_name = os.path.join(patch_path, str(i) + '.png')
                        io.imsave(p_name, patch_im)
                    else:
                        patch_im = np.expand_dims(patch_im, axis=-1)
                        if patch_im.shape[0]==64:
                            found = True
                            
                        patches_extracted = np.append(patches_extracted, patch_im, axis=-1)

                # save .npy file
                if self.save_file_type == '.npy':
                    p_name = os.path.join(patch_path, 'patchs.npy')
                    np.save(p_name, patches_extracted)
            else:  # volume image and per slice needed
                if self.save_file_type == '.npy':
                    shape = self.patch_dim[:2];
                    patches_extracted = np.empty(shape, 0)
                K = self.data.shape[-1]
                for k in range(K):
                    # read landmarks in kth slice
                    L = land_mark[-1]
                    L = L == k
                    I = np.nonzero(L)
                    landmark_kth = land_mark[:, :, I]
                    # kth slice
                    im = self.data.shape[:, :, k]

                    N = len(I)

                    for i in range(N):
                        R = int(landmark_kth[i, 0])
                        C = int(landmark_kth[i, 1])

                        check_coordinate = self.check_coordinate_patch(R, C)

                        # p = im[R - check_coordinate[0] : R + check_coordinate[1], C - check_coordinate[2]:C + check_coordinate[3]]
                        patch_im = im[check_coordinate[0]: check_coordinate[1],
                                   check_coordinate[2]:check_coordinate[3]]

                        # check file save type
                        if self.save_file_type == '.png':
                            # io.imsave(str(i)+'.png' , p)
                            # patch name
                            p_name = os.path.join(patch_path, str(i) + '.tiff')
                            io.imsave(p_name, patch_im)
                        else:
                            patches_extracted = np.append(patches_extracted, patch_im, axis=-1)
                # if self.save_file_type == '.npy':
                #     np.save('patches.npy',patches_extracted)
                if self.save_file_type == '.npy':
                    p_name = os.path.join(patch_path, 'patchs.npy')
                    np.save(p_name, patches_extracted)

        """         
         extract non tumor patches to extract non tumor patches we randomly selected a set of points in the image 
         and computed a distance from  center of tumor patches. If the selected patch center are at a distance 
         greatre than euclidean distance.  it is chosen as a center of non tumor patch. Search will continue until 
         a number of patches requested by the user are reached.  Number of rand points
         """

        num_non_tumor_selected = 0

        non_tumor_patch_center = np.empty((0, 2), 'uint16') # non tumor patchs center points

        # number of iteration for searching non-tumor patchs center points
        max_iter = 50
        iter_ = 0

        # compute ROI
        roi_im  = self.data > 0 *1

        #get row and column of pixels inside roi
        roi_row, roi_col = np.nonzero(roi_im)

        # Remove positions near boundary
        row_cond  = (roi_row > math.ceil(self.patch_dim[0]/2)) & (roi_row < (self.data.shape[0] - math.ceil(self.patch_dim[0]/2)))
        col_cond  = (roi_col > math.ceil(self.patch_dim[1]/2)) & (roi_col < (self.data.shape[1] - math.ceil(self.patch_dim[1]/2)))
        cond_both = row_cond & col_cond
        
        
        roi_row = roi_row[cond_both]

        roi_col = roi_col[cond_both]

        NN  = len(roi_row)

        while num_non_tumor_selected < self.num_nontumor_patch:
            # initial search space
            M = 3 * self.num_nontumor_patch;

            rand_ = random.sample(range(0, NN), M)
            rand_x = roi_row[rand_]
            rand_x = np.reshape(rand_x, (M, 1))
            rand_y = roi_col[rand_]
            rand_y = np.reshape(rand_y, (M, 1))
            rand_points = np.append(rand_x, rand_y, axis=1)

            # compute distance to tumor points
            eucl_dist = euclidean_distances(rand_points, landmarks_center)
            dist_thresh = max(self.patch_dim[1], self.patch_dim[0])

            eucl_dist = (eucl_dist > dist_thresh) * 1
            S = np.sum(eucl_dist, axis=1)

            # take the point if it is
            S = S == landmarks_center.shape[0]

            if np.sum(S) is not 0:
                points_needed = self.num_nontumor_patch - num_non_tumor_selected
                index = np.nonzero(S)[0]

                NN = min(len(index), points_needed)
                Indx = index[:NN]

                non_tumor_patch_center = np.append(non_tumor_patch_center, rand_points[Indx, :], axis=0)

                num_non_tumor_selected = num_non_tumor_selected + NN

            if iter_ > max_iter:
                break

            iter_ += 1

        # extract the patches
        # create empty matrix for npy file
        patch_path = os.path.join(path_, 'Normal')
        if os.path.exists(patch_path) is not True:
            os.mkdir(patch_path)
        if self.save_file_type == '.npy':
            patches_extracted = np.empty((self.patch_dim[0], self.patch_dim[1], 0), 'float32')

        for j in range(non_tumor_patch_center.shape[0]):
            R = non_tumor_patch_center[i, 0]  # row
            C = non_tumor_patch_center[i, 1]  # column

            # extract patch
            patch_im = self.data[R - math.ceil(self.patch_dim[0]/2): R + math.floor(self.patch_dim[0]/2),
                       C - math.ceil(self.patch_dim[1]/2): C + math.floor(self.patch_dim[1]/2)]

            # check file save type
            if self.save_file_type == '.png':
                # patch name
                p_name = os.path.join(patch_path, str(j) + '.png')
                io.imsave(p_name, patch_im)
            else:
                patch_im_exp= np.expand_dims(patch_im, axis=-1)
                print(patch_im_exp.shape)
#                if patch_im.shape[0] >32:
#                    found = True
                patches_extracted = np.append(patches_extracted, patch_im_exp, axis=-1)
                
        if self.save_file_type == '.npy' and num_non_tumor_selected is not 0:
            p_name = os.path.join(patch_path, 'patchs.npy')
            np.save(p_name, patches_extracted)



    def check_coordinate_patch(self,R,C, D=None):

        """
        THis function takes center point of a patch and check if a patch can be fiied with the requested size can be fit.
        and return corrected one
        R----row
        C....column
        D.....Depth
        """

        row_half_t= math.ceil(self.patch_dim[0]/2)
        row_half_b= math.floor(self.patch_dim[0]/2)
        
        col_half_l= math.ceil(self.patch_dim[1]/2)
        col_half_r= math.floor(self.patch_dim[1]/2)

        if (R - row_half_t) <0:
            R1 = 0
            R2 = self.patch_dim[0]

        elif (R + row_half_b) >self.data.shape[0]:
            R2 = self.data.shape[0]
            R1 = R2 - self.patch_dim[0]
        else:
            R1 = R - row_half_t
            R2 = R + row_half_b
         # check column
        if (C - col_half_l) <0:
            C1 = 0
            C2 = self.patch_dim[1]
        elif (C + col_half_r) >self.data.shape[1]:
            C2 = self.data.shape[1]
            C1 = C2 - self.patch_dim[1]
        else:
            C1  = C - col_half_l
            C2  = C + col_half_r

        if D is not None and D is not 0: #3D point
            depth_half_f= math.ceil(self.patch_dim[2]/2) # front half
            depth_half_b= math.floor(self.patch_dim[2]/2) #back half
            if (D - depth_half_f) <0:
                D1 = 0
                D2 = self.patch_dim[2]
            elif (D + depth_half_b) >self.data.shape[2]:
                D2 = self.data.shape[2]
                D1 = D2 - self.patch_dim[2]
            else:
                D1 = D - depth_half_f
                D2 = D + depth_half_b
            check_coordinate = (R1 , R2 , C1, C2, D1, D2)
            
        else:
            # R1  = R - row_half_t
            # R2  = R + row_half_b
            # C1  = C - col_half_l
            # C2  = C + col_half_r
            check_coordinate = (R1 , R2 , C1, C2)
            
            
        return check_coordinate
    
    def extractPatch_3D_patchs_mask_gt(self):

        """
        This function it extarcts 3D patches using mask or ground truth or ROI image into consideration
        """

        if self.appply_on_batch_files is False:  # single image
            # get image name and extension
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]



            # for gt image
            gt_image_name = self.gt_img_path
            # get image extension
            gt_image_name = os.path.split(gt_image_name)
            gt_image_name = gt_image_name[-1]
            gt_img_name_split = gt_image_name.split(".")
            gt_img_extension = gt_img_name_split[-1]

            name_ = os.path.join(self.gt_img_path)
            if gt_img_extension == 'dcm':
                self.gt_data = read_dicom(name_)
            elif gt_img_extension == 'nii' or gt_img_extension == 'gz' or gt_img_extension == 'nii.gz':
                self.gt_data = read_nii(name_)
            else:
                self.gt_data = read_image(name_)

            # convert to 8-bit

            self.data = self.to_8bit_(self.data)

            # extract patch
            im_patches, gt_patches = extract_3d_patches(self.data , self.patch_dim,self.stride, gt_data=self.gt_data)

            # save patch
            path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

            np.save(path_, im_patches)

            path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

            np.save(path_, gt_patches)

        else: # for batch of files
            assert len(os.listdir(self.img_source_folder)) or \
                   len(os.listdir(self.gt_path_source_folder)), 'source folder is empty'
            assert len(os.listdir(self.gt_path_source_folder)) == len(os.listdir(self.img_source_folder)), \
                'number of xml file and image files shoud be same, BUT, found different'

            img_files = os.listdir(self.img_source_folder)
            gt_files = os.listdir(self.gt_path_source_folder)
            for images_name, gt_name in zip(img_files, gt_files):
                self.image_name = images_name
                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                name_ = os.path.join(self.img_source_folder, images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)

                # convert to 8-bit
                self.data = self.to_8bit_(self.data)

                # for gt image
                gt_image_name = gt_name
                # get image extension
                gt_image_name = os.path.split(gt_image_name)
                gt_image_name = gt_image_name[-1]
                gt_img_name_split = gt_image_name.split(".")
                gt_image_name = gt_img_name_split[0]
                gt_img_extension = gt_img_name_split[-1]

                name_ = os.path.join(self.gt_path_source_folder, gt_name)
                if gt_img_extension == 'dcm':
                    self.gt_data = read_dicom(name_)
                elif gt_img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.gt_data = read_nii(name_)
                else:
                    self.gt_data = read_image(name_)

                # extract 3D patches

                im_patches, gt_patches = extract_3d_patches(self.data, self.patch_dim, self.stride, gt_data=self.gt_data)

                # Save patches
                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                np.save(path_, im_patches)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

                np.save(path_, gt_patches)


        print('Patch Generation done!')


    def extractPatch_3D_patchs_no_gt(self):
        """
        Extract 3D patches from 3D image; when there is no ground truth provided.
        """

        if self.appply_on_batch_files is False:  # single image
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]

            # change to 8 -bit
            self.data = self.to_8bit_(self.data)

            # extarct and save images

            im_patches = extract_3d_patches(self.data , self.patch_dim, self.stride, gt_data= None)

            path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

            np.save(path_, im_patches)

        else: # for batch of files
            assert len(os.listdir(self.img_source_folder)), 'source folder is empty'

            img_files = os.listdir(self.img_source_folder)
            for images_name in img_files:
                self.image_name = images_name

                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                name_ = os.path.join(self.img_source_folder, images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)

                # convert to 8-bit
                self.data = self.to_8bit_(self.data)

                # extarct and save images

                im_patches, gt_patches = extract_3d_patches(self.data, self.patch_dim, self.stride, gt_data= None)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                np.save(path_, im_patches)

        print('Patch Generation done!')


    def extractPatch_2D5_patchs_no_gt(self):
        """
        Extract 2.5D patches from 3D image; when there is no ground truth provided.
        """

        patch_size_temp = (self.patch_dim[0], self.patch_dim[0], self.patch_dim[0])

        if self.appply_on_batch_files is False:  # single image
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]

            self.data = self.to_8bit_(self.data)

            im_patches = extract_3d_patches(self.data , patch_size_temp, self.stride, gt_data= None)

            output_patches = self.make_2D5_patch_from_3d(im_patches)

            path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

            np.save(path_, output_patches)

        else: # for batch of files
            assert len(os.listdir(self.img_source_folder)), 'source folder is empty'

            img_files = os.listdir(self.img_source_folder)
            for images_name in img_files:
                self.image_name = images_name

                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                name_ = os.path.join(self.img_source_folder, images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)

                # convert to 8-bit
                self.data = self.to_8bit_(self.data)

                # extract and save patchs

                im_patches = extract_3d_patches(self.data, patch_size_temp, self.stride, gt_data= None)

                output_patches = self.make_2D5_patch_from_3d(im_patches)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                np.save(path_, output_patches)


        print('Patch Generation done!')

    def extractPatch_2D5_patchs_mask_gt(self):
        """
           Extract 2.5D patches from 3D image with ground truth image provided.
           """
        # the size of the patch will only depend on the first patchg dimension
        patch_size_temp = (self.patch_dim[0], self.patch_dim[0], self.patch_dim[0])

        if self.appply_on_batch_files is False:  # single image
            # get image name and extension
            img_name = self.image_name
            img_name = os.path.split(img_name)
            img_name = img_name[-1]
            img_name_split = img_name.split(".")
            self.img_name_ = img_name_split[0]

            # for gt image
            gt_image_name = self.gt_img_path
            # get image extension
            gt_image_name = os.path.split(gt_image_name)
            gt_image_name = gt_image_name[-1]
            gt_img_name_split = gt_image_name.split(".")
            gt_img_extension = gt_img_name_split[-1]

            name_ = os.path.join(self.gt_img_path)
            if gt_img_extension == 'dcm':
                self.gt_data = read_dicom(name_)
            elif gt_img_extension == 'nii' or gt_img_extension == 'gz' or gt_img_extension == 'nii.gz':
                self.gt_data = read_nii(name_)
            else:
                self.gt_data = read_image(name_)

            # convert to 8 bit
            self.data = self.to_8bit_(self.data)

            # extract and  save patchs

            im_patches, gt_patches = extract_3d_patches(self.data, patch_size_temp, self.stride, gt_data=self.gt_data)

            output_patches = self.make_2D5_patch_from_3d(im_patches)

            path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

            np.save(path_, output_patches)

            output_patches_gt = self.make_2D5_patch_from_3d(gt_patches)

            path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

            np.save(path_, output_patches_gt)

        else:  # for batch of files
            assert len(os.listdir(self.img_source_folder)) or \
                   len(os.listdir(self.gt_path_source_folder)), 'source folder is empty'
            assert len(os.listdir(self.gt_path_source_folder)) == len(os.listdir(self.img_source_folder)), \
                'number of xml file and image files shoud be same, BUT, found different'

            img_files = os.listdir(self.img_source_folder)
            gt_files = os.listdir(self.gt_path_source_folder)
            for images_name, gt_name in zip(img_files, gt_files):
                self.image_name = images_name
                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                self.img_name_ = img_name_split[0]
                img_extension = img_name_split[-1]

                name_ = os.path.join(self.img_source_folder, images_name)
                if img_extension == 'dcm':
                    self.data = read_dicom(name_)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.data = read_nii(name_)
                else:
                    self.data = read_image(name_)

                # convert to 8-bit
                self.data = self.to_8bit_(self.data)

                # for gt image
                gt_image_name = gt_name
                # get image extension
                gt_image_name = os.path.split(gt_image_name)
                gt_image_name = gt_image_name[-1]
                gt_img_name_split = gt_image_name.split(".")
                gt_image_name = gt_img_name_split[0]
                gt_img_extension = gt_img_name_split[-1]

                name_ = os.path.join(self.gt_path_source_folder, gt_name)
                if gt_img_extension == 'dcm':
                    self.gt_data = read_dicom(name_)
                elif gt_img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.gt_data = read_nii(name_)
                else:
                    self.gt_data = read_image(name_)

                # extract and  save patchs

                im_patches, gt_patches = extract_3d_patches(self.data, patch_size_temp, self.stride,
                                                            gt_data=self.gt_data)

                output_patches = self.make_2D5_patch_from_3d(im_patches)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_images_.npy')

                np.save(path_, output_patches)

                output_patches_gt = self.make_2D5_patch_from_3d(gt_patches)

                path_ = os.path.join(self.generated_patches_folder, self.img_name_ + '_gt_.npy')

                np.save(path_, output_patches_gt)

        print('Patch Generation done!')


    def make_2D5_patch_from_3d(self, volume):
        """
            This function accepts a 3D VOLUME patch and remove some part to create 2.5D patches
            it taks the axial, cornal and sagital slices only
        """
        # nuymber of voumes
        N  = volume.shape[0]
        out_patch = np.zeros((N,self.patch_dim[0], self.patch_dim[0],3))
        C = math.ceil(self.patch_dim[0]/2)

        for i in range(N):
            X  = volume[i,:,:,:]
            out_patch[i, :, :, 0] = X[C,:,:] # axial
            out_patch[i, :, :, 1] = X[:, :, C] # cornal
            out_patch[i, :, :, 2] = X[:, C,:] # sagital
        return out_patch

    def to_8bit_(self, image_in):
        """
        It accepts an image and it converts it to 8 bit image

        :param image_in: Input image
        :return: 8-bit image
        """
        # change image to 8 bit
        min_ = np.min(image_in)
        max_ = np.max(image_in - min_)
        image_in = (image_in - min_).astype('float') / float(max_) * 255

        return  image_in.astype('uint8')
