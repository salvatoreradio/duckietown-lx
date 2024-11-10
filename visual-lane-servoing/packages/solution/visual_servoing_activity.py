from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    
    steer_matrix_left = np.zeros(shape)
    shape = [shape[0], int(shape[1]/2)]
    for i in range(shape[1]):
        for j in range(shape[0]-i-1, shape[0]):
            steer_matrix_left[j][i] = -1
            
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    
    steer_matrix_right = np.zeros(shape)
    shape_temp = [shape[0], int(shape[1]/2)]
    temp_matrix = steer_matrix_right[:, shape_temp[1]:]
    
    for i in range(shape_temp[0]):
        for j in range(0, (i+1)+(shape_temp[1]-shape_temp[0])):
              if j < shape_temp[1]:
                temp_matrix[i][j] = 1
                
    steer_matrix_right[:, shape_temp[1]:] = temp_matrix
            
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    # TODO: implement your own solution here
    
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask_ground = np.ones(image.shape, dtype=np.uint8)
    mask_ground[0:200, :] = 0
    
    sigma = 5
    img_gaussian_filter = cv2.GaussianBlur(image,(0,0), sigma)
    
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)
    
    threshold = 50
    mask_mag = (Gmag > threshold)
    
    white_lower_hsv = np.array([0, 0, 135])
    white_upper_hsv = np.array([120, 70, 255])
    yellow_lower_hsv = np.array([20, 75, 120])
    yellow_upper_hsv = np.array([30, 255, 225])

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0
    
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)
    
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white
    
    # mask_left_edge = np.random.rand(h, w)
    # mask_right_edge = np.random.rand(h, w)

    return mask_left_edge, mask_right_edge
