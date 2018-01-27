#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:55:53 2017

@author: yeman and minh
"""
import sys
import os

# image display in Qlabel
import qimage2ndarray

# image processing
import  cv2


from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.uic import loadUi

# local packages
from GeneratePatch.GeneratePatch import GeneratePatch
from AugmentImage.AugmentImage import  AugmentImage
from Utilits.xml_utilts import get_type_center_from_XML
from Utilits.image_reading_utilits import read_dicom,read_image, read_nii

from other_preprocessing.OtherPreprocessing import OtherPreprocessing

class ImageAugmentPatchExtact(QMainWindow):
    """
    main window class
    """
    def __init__(self):
        """
        Class constructor

        initialize window parameters
        """

        super(ImageAugmentPatchExtact, self).__init__()
        # load Ui
        loadUi('Ui/ImageAugmentPatchExtact.ui',self)
        # variable to hold any image_data loaded
        self.image_data  = None
        self.land_mark  = None
        self.image_name = None
        self.patch_saving_folder = None
        self.augmentation_saving_folder = None
        
        # histogram atching
        self.ref_image = None
        
        self.appply_on_batch_files = False
        self.adaptive_histeq = False
        self.source_folder = None

        # augmentation
        self.landmark_source_folder = None

        # common
        self.dst_folder = None
        self.xml_path  = None

        # patch generation
        self.stride  = 32
        self.gt_img_path  = None
        self.gt_path_source_folder = None

        # for window
        self.title = 'Image Preparation for Machine Learning'

        # initialize Ui
        self.initUI()

    def initUI(self):

        # status bar message
        self.statuslabel.setText('Waiting')

        # window setting
        self.setWindowTitle(self.title)
        self.setGeometry(500, 200, 1000, 600)

        ## ==============signals and slots===================================


        # Visualization section
        #slider
        self.slider.valueChanged.connect(self.slidervaluechanged)


        #%%patch extraction section
        
        # buttuns of extact patches
        self.patch_generation_load_btn.clicked.connect(self.patch_generation_load_img_btn_clicked)
        self.load_landmarks_btn.clicked.connect(self.patchgen_load_landmarks_btn_clicked)
        self.extract_patches_btn.clicked.connect (self.extractPatches)
        self.select_patch_saving_folder_btn.clicked.connect(self.select_destination_folder)
        self.gt_mask_btn.clicked.connect(self.patchgen_load_gt_images_btn_clicked)

        # radiobtn section
        self.landmark_gt_radioButton.toggled.connect(self.groundtruth_type_radio_btn_state_changed)
        self.mask_gt_radioButton.toggled.connect(self.groundtruth_type_radio_btn_state_changed)
        self.none_gt_radioButton.toggled.connect(self.groundtruth_type_radio_btn_state_changed)


        #%% Augmentation section
        
        # CheckBox
        self.load_aug_landmark_checkBox.stateChanged.connect(self.load_aug_landmark_checkBox_checked)

        #buttons of augmentation
        self.aug_load_img_btn.clicked.connect(self.aug_load_img_btn_clicked)
        self.aug_load_landmarks_btn.clicked.connect(self.aug_load_landmarks_btn_clicked)
        self.aug_dst_folder_btn.clicked.connect(self.select_destination_folder)
        self.aug_apply_btn.clicked.connect (self.apply_agugmentation)


        # Other preprocessing section

        # CheckBox
        self.hist_match_checkBox.stateChanged.connect(self.hist_match_checkBox_checked)

        # radio btn
        self.histogram_equa_radio_btn.toggled.connect(self.histeq_radio_btn_state)
        self.adap_histogram_equa_radio_btn.toggled.connect(self.histeq_radio_btn_state)


        # buttons of other preproceessing
        self.otherpreprocessing_load_img_btn.clicked.connect(self.otherpreprocessing_load_img_btn_clicked)
        self.otherpreprocessing_apply_btn.clicked.connect(self.other_preprocessing_apply_btn_clicked)
        self.otherpreprocessing_dst_folder_btn.clicked.connect(self.select_destination_folder)
        self.load_ref_img_btn.clicked.connect(self.load_reference_image)


    #*******************************************************************************************************************

    # Patch Generation Methods Section

    # ***************************************************************************************************************


    def patch_generation_load_img_btn_clicked(self):

        self.loadImageimage_data(apply_to_folder_files=self.apply_to_folder_patchextratcion_checkBox.isChecked())

    def patchgen_load_landmarks_btn_clicked(self):
        self.loadLandmarks(apply_to_folder_files=self.apply_to_folder_patchextratcion_checkBox.isChecked())
    
    def patchgen_load_gt_images_btn_clicked(self):
        self.load_gt_images(apply_to_folder_files=self.apply_to_folder_patchextratcion_checkBox.isChecked())


    def groundtruth_type_radio_btn_state_changed(self):
        if self.landmark_gt_radioButton.isChecked() is True:
            self.load_landmarks_btn.setEnabled(True)
            self.gt_mask_btn.setEnabled(False)
            # stride_spinBox
            self.stride_spinBox.setEnabled(False)

        elif self.mask_gt_radioButton.isChecked() is True:
            self.load_landmarks_btn.setEnabled(False)
            self.gt_mask_btn.setEnabled(True)
            # stride_spinBox
            self.stride_spinBox.setEnabled(True)

        elif self.none_gt_radioButton.isChecked() is True:
            self.load_landmarks_btn.setEnabled(False)
            self.gt_mask_btn.setEnabled(False)
            # stride_spinBox
            self.stride_spinBox.setEnabled(True)


    def get_groundtruth_type_radio_btn_state(self):
        if self.landmark_gt_radioButton.isChecked() is True:
            self.ground_truth_type  = 'landmark'
        elif self.mask_gt_radioButton.isChecked() is True:
            self.ground_truth_type = 'gt_mask'

        elif self.none_gt_radioButton.isChecked() is True:
            self.ground_truth_type = None

    def load_gt_images(self, apply_to_folder_files = False):

        if apply_to_folder_files is True:
            self.gt_path_source_folder = str(QFileDialog.getExistingDirectory(self, "Select a directory containing ground truth files"))

            if os.path.isdir(self.gt_path_source_folder ) is False or len(os.listdir(self.gt_path_source_folder)) is 0:
                QMessageBox.warning(self, "Warning",
                                    "Please select a folder again or the directory is empty")
            else:
                self.gt_img_path = None

        else:
            filter = "All Files (*);;image files (*.nii.gz *.png *.tif *.PNG *.dcm)"
            title = "Select image or folder"
            self.gt_img_path = self.openFileNameDialog(filter, title)

            # lmark = get_type_center_from_XML(self.xml_path)
            #
            # self.land_mark = lmark




    def extractPatches(self):

        # get all patch parametes

        # patch details
        patch_row = self.patch_row.value()
        patch_col = self.patch_col.value()
        patch_depth = self.patch_depth.value()
        patch_type_comboBox = self.patch_type_comboBox.currentText()
        patch_save_file_type_comboBox = self.p_save_file_type_comboBox.currentText()
        patch_size = (patch_row, patch_col, patch_depth)
        
        # number of tumor and non tumor patches
        
        num_tumor_patches  = self.num_tumor_patches.value()
        num_nontumor_patches  = self.num_ntumor_patches.value()

        self.get_groundtruth_type_radio_btn_state()

        # get stride value
        if self.none_gt_radioButton.isChecked() is True or self.mask_gt_radioButton.isChecked() is True:
            self.stride  = self.stride_spinBox.value()
        
        # instantiate patch generation class
        
        patch_generator  = GeneratePatch(data=self.image_data,
                                         name = self.image_name,
                                         landmark = self.land_mark,
                                         appply_on_batch_files=self.apply_to_folder_patchextratcion_checkBox.isChecked(),
                                         img_source_folder=self.source_folder,
                                         landmark_source_folder=self.landmark_source_folder,
                                         landmark_path=self.xml_path,
                                         patch_type = patch_type_comboBox, 
                                         patch_dim = patch_size,
                                         patch_percntg = None,# to be considered in future
                                         num_tumor_patch = num_tumor_patches, 
                                         num_nontumor_patch = num_nontumor_patches, 
                                         save_file_type = patch_save_file_type_comboBox ,
                                         dst_folder = self.dst_folder,

                                         gt_path_source_folder = self.gt_path_source_folder,
                                         gt_img_path  = self.gt_img_path,
                                         stride  = self.stride,
                                         ground_truth_type  = self.ground_truth_type)
        # update status
        self.statuslabel.setText('Running patch extracting....')
        
        # extract patch
        patch_generator.patch_generator_main()

        # update status
        self.statuslabel.setText('Patch extraction finished. please see the destination folder!')

    #*******************************************************************************************************************

    # Augmentation Methods Section

    # *******************************************************************************************************************

    def aug_load_img_btn_clicked(self):
        self.loadImageimage_data(apply_to_folder_files=self.apply_to_folder_aug_checkBox.isChecked())

    def aug_load_landmarks_btn_clicked(self):
        self.loadLandmarks(apply_to_folder_files=self.apply_to_folder_aug_checkBox.isChecked())


    def load_aug_landmark_checkBox_checked(self):
        if self.load_aug_landmark_checkBox.isChecked() is True:
            self.aug_load_landmarks_btn.setEnabled(True)
        else:
            self.aug_load_landmarks_btn.setEnabled(False)
            
    def apply_agugmentation(self):
        # get rotation angle
        rotation_angle = self.rotation_angle_spinBox.value()

        AugmentImage_obj = AugmentImage(data = self.image_data,
                                        img_name= self.image_name,
                                        appply_on_batch_files=self.apply_to_folder_aug_checkBox.isChecked(),
                                        img_source_folder=self.source_folder,
                                        landmark_source_folder = self.landmark_source_folder,
                                        landmark_path  = self.xml_path,
                                        rotation_angle = rotation_angle,
                                        dst_folder = self.dst_folder,
                                        flip_vertical = self.flip_vertical_checkBox.isChecked(),
                                        flip_horizental=self.flip_horizontal_checkBox.isChecked()
                                        )

        self.statuslabel.setText('Running image augmentation...')
        AugmentImage_obj.apply_augmentation()

        self.statuslabel.setText('Image augmentation finished. Please see the destination folder!')


    #*******************************************************************************************************************

    # Other Preprocessing Methods Section

    # ****************************************************************************************************************

    def otherpreprocessing_load_img_btn_clicked(self):
        self.loadImageimage_data(apply_to_folder_files=self.apply_to_folder_otherprepro_checkBox.isChecked())

    def hist_match_checkBox_checked(self):
        if self.hist_match_checkBox.isChecked() is True:
            self.load_ref_img_btn.setEnabled(True)
        else:
            self.load_ref_img_btn.setEnabled(False)


    def load_reference_image(self):
        filter = "All Files (*);;image files (*.nii.gz *.png *.tif *.PNG *.dcm)"
        title =  "Select refernce image"
        ref_image_name = self.openFileNameDialog(filter, title)

        # self.ref_image  = io.imread(image_name)
        # get image extension
        img_name = ref_image_name
        img_name = os.path.split(img_name)
        img_name = img_name[-1]
        img_name_split = img_name.split(".")
        img_extension = img_name_split[-1]

        if img_extension == 'dcm':
            self.ref_image = read_dicom(ref_image_name)
        elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
            self.ref_image = read_nii(ref_image_name)
        else:
            self.ref_image = read_image(ref_image_name)



    def histeq_radio_btn_state(self):
        if self.adap_histogram_equa_radio_btn.isChecked() is True:
            self.adaptive_histeq  = True
        else:
            self.adaptive_histeq = False
    def load_ref_img_btn(self):
        pass


    def other_preprocessing_apply_btn_clicked(self):
        self.histeq_radio_btn_state()

        otherPreprocessing_obj = OtherPreprocessing(input_image  = self.image_data,
                                                    ref_image = self.ref_image,
                                                    appply_on_batch_files  = self.apply_to_folder_otherprepro_checkBox.isChecked(),
                                                    source_folder=self.source_folder,
                                                    dst_folder= self.dst_folder,
                                                    img_name= self.image_name,
                                                    normalize=self.normalization_checkBox.isChecked(),
                                                    hist_match = self.hist_match_checkBox.isChecked(),
                                                    adaptive_histeq = self.adaptive_histeq)
        

        # update status

        self.statuslabel.setText('Running image augmentation...')

        # apply preprocessing
        otherPreprocessing_obj.apply_otherpreprocessing()

        #update status
        self.statuslabel.setText('Image augmentation finished. Please see the destination folder!')



    #*******************************************************************************************************************

    # Common Methods Section

    # ****************************************************************************************************************


    def loadImageimage_data(self, apply_to_folder_files = False):

        if apply_to_folder_files is True:
            self.source_folder = str(QFileDialog.getExistingDirectory(self, "Select a directory containing images"))

            if os.path.isdir(self.source_folder) is False or len(os.listdir(self.source_folder)) is 0:
                QMessageBox.warning(self, "Warning",
                                    "Please select a folder or the directory is empty")
            else:
                self.image_data = None
                self.image_name = None
        else:
            filter = "All Files (*);;image files (*.nii.gz *.png *.tif *.PNG *.dcm)"
            title = "Select image or folder"
            self.image_name = self.openFileNameDialog(filter, title)

            if os.path.isdir(self.image_name) is True:
                print('you have selected a folder, please select a file!')
            else:

                # get image extension
                img_name = self.image_name
                img_name = os.path.split(img_name)
                img_name = img_name[-1]
                img_name_split = img_name.split(".")
                img_extension = img_name_split[-1]

                # allowed image formats
                image_format = ('.png', '.PNG', '.tif', '.dcm', '.nii', 'nii.gz')

                assert  self.image_name.endswith(image_format) is True, 'selected file doesnot  have supported image extension. allowed image ' \
                                                                        'image extensions are (.png, .PNG, .tif, .dcm, .nii, nii.gz)'

                if img_extension == 'dcm':
                    self.image_data = read_dicom(self.image_name)
                elif img_extension == 'nii' or img_extension == 'gz' or img_extension == 'nii.gz':
                    self.image_data = read_nii(self.image_name)
                else:
                    self.image_data = read_image(self.image_name)


                # display image

                # imaage shape
                S = self.image_data.shape
                if len(S) == 2: # 2d image
                    self.slider.setMinimum(0)
                    self.slider.setMaximum(0)
                    self.displayNDArray(self.image_data)

                else: # 3D image
                    N = S[-1]

                    self.slider.setMinimum(0)
                    self.slider.setMaximum(N - 1)
                    self.slider.setValue(int(N / 2))
                    self.slider.setTickInterval(1)

                    self.displayNDArray(self.image_data[:, :, int(N / 2)])

    def loadLandmarks(self, apply_to_folder_files = False):

        if apply_to_folder_files is True:
            self.landmark_source_folder = str(QFileDialog.getExistingDirectory(self, "Select a directory containing xml files"))

            if os.path.isdir(self.landmark_source_folder) is False or len(os.listdir(self.landmark_source_folder)) is 0:
                QMessageBox.warning(self, "Warning",
                                    "Please select a folder again or the folder is empty")
            else:
                self.land_mark = None
                #self.image_name = None
        else:
            filter = "xml files (*.xml)"
            title = "Select XML file"
            self.xml_path = self.openFileNameDialog(filter, title)

            assert self.xml_path.endswith('xml')  is True, 'the selected file is not xml file, only xml file allowed!'

            lmark = get_type_center_from_XML(self.xml_path)

            self.land_mark = lmark


    def displayNDArray(self, image):


        # scale the image to size smaller than label sothat it will be displayed QLabel
        # this can any scale, finally image will be scaled to Qlable size
        image = cv2.resize(image, (400, 400))

        # Convert to QImage
        pixmap  = qimage2ndarray.array2qimage(image, normalize = True)
        qimg = QPixmap.fromImage(pixmap)
        self.label1.setPixmap(qimg)
        self.label.setScaledContents(True)


    def slidervaluechanged(self):
        # vsualize next volume
        value = self.slider.value()
        self.displayNDArray(self.image_data[:,:,value])


    def openFileNameDialog(self, filter, title):
        # open file dialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, title, "", filter, options=options)
        return fileName


    def select_destination_folder(self):

        self.dst_folder = str(QFileDialog.getExistingDirectory(self, "Select destination directory"))

        if os.path.isdir(self.dst_folder) is False:
            QMessageBox.warning(self, "Warning",
                                "Please select a folder")


if __name__ == '__main__':
    """
    main function for luanching the software!
    """
    app = QApplication(sys.argv)
    widget = ImageAugmentPatchExtact()
    widget.show()
    sys.exit(app.exec_())

