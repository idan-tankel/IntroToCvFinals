from operator import le
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image


def generate_point_correspondence():
    src_points = np.array([(1,2,3)])
    dst_points = np.array([(1,1,1)])
    return src_points,dst_points


def create_the_matrix(src_points: np.array, dst_points: np.array) -> np.array:
    """
    Creates the matrix A for which we will have
    Ae = 0 for e=(e₁₁....e₃₃)
    Args
    -----
        `src_points` - list of points from the first image. each point should be an array of numpy
        `dst_points` - list of points (tuple of integers - pixel location) from the second image, correspoding the first list
    foreach i, 
    src_points[i]*E*dst_points[i] = 0
    """
    # we will build the matrix iteratively, as we did within the 
    assert len(src_points) == len(dst_points),"The lists should be a match of pairs - and hold the sampe length"
    A = []
    number_of_pairs = len(src_points)
    for i in range(0,number_of_pairs):
        multiplication = (src_points[i].reshape(3,1))*dst_points[i]
        A.append(multiplication.reshape(-1)) # trick -1 is reshape to 1d array
    return A


def solve_by_svd(A:np.array):
    """
    This function gets the matrix A from the `create_the_matrix` function and apply SVD to it
     in order to find the H params

    Args:
        A: numpy.array - The matrix to apply SVD to
    """
    U, S, V = np.linalg.svd(A)
    h = V[-1,:] / V[-1,-1] # this is normalization trick - for the last column, V[-1], apply to all the coordinates V[-1,:]....
    # H is now the vector of solutions
    h = h.reshape(3,3)
    h = np.round(h,2)
    # todo - normalization!

    return h
