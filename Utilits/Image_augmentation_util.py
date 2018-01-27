import cv2
import numpy as np

def get_rotation_maxtrix(image,angle):
    """
    generate transformation matrix and image and outputt image dimension
    :param image: input image
    :param angle: rotation angle
    :return: Transformation angle and output image dimension
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    im_dim = (nW,nH)

    return  M,im_dim

def rotate_image(image, M, output_dim):
    """
    Rotate an image
    :param image: input image
    :param M: transformation matrix
    :param output_dim: expected output image dimension
    :return: roatetd image
    """
    assert (len(image) is not 0 or len(output_dim) is not 0 or len(M) is not 0), \
        'landmark_points and/or rotation matrix cannot be empty'

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, output_dim)



def rotate_landmark(landmark_points, M):
    """
    Apply transformation matrix on laandmark points
    :param landmark_points: input landmark points
    :param M:transformation matrix
    :return:transformed landmarks
    """
    assert( len(M) is not 0 or len(landmark_points) is not 0 ),\
        'landmark_points and/or rotation matrix cannot be empty'

    # apply transformation on the points
    landmark_points  = landmark_points.T
    landmark_points  = np.append(landmark_points, np.ones((1, landmark_points.shape[1])))

    T_landmark_points = np.matmul(M,landmark_points)

    return  T_landmark_points


