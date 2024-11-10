from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    
    weights_matrix = np.zeros((480, 640))
    weights_matrix[320:, :] = 1.0
    weights_matrix[200:320, :] = 0.5
    weights_matrix[100:200, :] = 0.1

    
    res[100:200, 100:320] = -1
    res[200:480, 100:320] = 1
    res[200:480, 320:540] = -1
    res[100:200, 320:540] = 1
    
    final = res * weights_matrix

    # ---
    return final


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values

    weights_matrix = np.zeros((480, 640))
    weights_matrix[320:, :] = 1.0
    weights_matrix[200:320, :] = 0.5
    weights_matrix[100:200, :] = 0.1
    
    res[100:200, 100:320] = 1
    res[200:480, 100:320] = -1
    res[200:480, 320:540] = 1
    res[100:200, 320:540] = -1
    
    final = res * weights_matrix
    
    # ---
    return final
