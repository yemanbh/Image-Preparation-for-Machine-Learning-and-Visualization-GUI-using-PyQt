#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:09:34 2017

@author: yb
"""

import xml.etree.ElementTree as ET
import numpy as np
from Utilits.Image_augmentation_util import rotate_landmark

def get_type_center_from_XML(xml_path):
    """
    This function searchs contour poins, compute their center and return them as dictionary
    :param xml_path:xml file path
    :return:dictionary of all abnormalities center point
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    points_dict = {}
#    dict_num = 0
    for elem in root:
        for arry_elem1 in elem.findall('array'):
            # print(image_elem.text)
            for dict_elem1 in arry_elem1:
                for arry_elem2 in dict_elem1.findall('array'):
                    for arry_elem2_dict in arry_elem2:
                        # point_dict = {}
                        name_tag_found = False
                        Point_px_tag_found = None
                        center_x = 0
                        center_y = 0
                        for ROI_detail in arry_elem2_dict:

                            # print(ROI_detail.text)


                            if name_tag_found is True:
                                # get point catagory
                                # this tag is type of finding (mass, calcification, distortion,spiculated region)
                                point_type  = ROI_detail.text
                                # point_dict['type'] =
                                name_tag_found = False

                            if Point_px_tag_found is True:
                                # get center point
                                num_contour_point = len(ROI_detail)
                                for point_px_elem in ROI_detail:
                                    P = point_px_elem.text
                                    split = P.split()
                                    X = float(split[0][1:len(split[0]) - 1])
                                    center_x += X
                                    Y = float(split[1][:len(split[1]) - 1])
                                    center_y += Y
                                Point_px_tag_found = False
                                center_x = round(center_x / num_contour_point)
                                center_y = round(center_y / num_contour_point)
                                center_coord = np.array([center_x, center_y])
                                center_coord = np.reshape(center_coord,[1,len(center_coord)])

                                if point_type in points_dict:
                                    # append the center point
                                    points  = points_dict[point_type]
                                    points_dict[point_type] = np.append(points,center_coord,axis=0)

                                else:
                                    #create another key
                                    points_dict[point_type] = center_coord

                            if ROI_detail.text == 'Name':
                                # next point is catagory of the point
                                name_tag_found = True

                            if ROI_detail.text == 'Point_px':
                                # Next tag is list of contour points
                                Point_px_tag_found = True


    print('done!')
    return points_dict


def update_xml(xml_tree, rotation_matrix):
    """
    This function apply rotation to contour points in XML files
    :param xml_tree: XML tree object
    :param rotation_matrix: Transformation matrix
    :return: modified XML tree object
    """
    root = xml_tree.getroot()
    for string_tag in root.iter('string'):
        string_tag_val = string_tag.text
        print(string_tag_val)
        if len(string_tag_val.split()) > 1:
            splited_str = string_tag_val.split()
            print(splited_str[0][1:len(splited_str[0]) - 1])
            X  = splited_str[0][1:len(splited_str[0]) - 1];
            X  = X.strip()
            X = float(X)
            Y  = splited_str[1][:len(splited_str[1]) - 1];
            Y = Y.strip()
            Y = float(Y)
            contour_point = np.array([X, Y])
            contour_point = np.reshape(contour_point, [1, len(contour_point)])
            transformed_contour_points = rotate_landmark(contour_point,rotation_matrix)
            
            if len(string_tag_val.split()) == 2:
                string_tag.text = '(' + str(transformed_contour_points[0]) + ', ' + str(transformed_contour_points[1]) + ')'
            else:
                string_tag.text = '(' + str(transformed_contour_points[0]) + ', ' + str(transformed_contour_points[1]) + ', ' + str(splited_str[2]) + ')'

            # updated values

            X_ = float('{:f}'.format(transformed_contour_points[0]))
            Y_ = float('{:f}'.format(transformed_contour_points[1]))

            if X_< 0:
                X_ = 0.0000

            if Y_ < 0:
                Y_ = 0.0000

            if len(string_tag_val.split()) == 2:
                string_tag.text = '(' + str(X_) + ', ' + str(Y_) + ')'
            else:
                string_tag.text = '(' + str(X_) + ', ' + str(Y_) + ', ' + str(splited_str[2]) + ')'
            
                

    return xml_tree



