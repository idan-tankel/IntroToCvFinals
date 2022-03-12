from operator import le
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
# class homography():

def apply_homography(image):
    """
    this function is applying a simple homography on an image
    Args:
        imgae: the PIL image to apply homography on
    """

    with Image.open(r'./data/book1.png') as image_object:
        alpha = math.pi / 15.
        new_object = image_object.transform(data=(
            math.cos(alpha), math.sin(alpha), 20,
            -math.sin(alpha), math.cos(alpha), 20,
            ),method=Image.AFFINE,size=(1000,1000)
        )
        plt.imshow(new_object)
    return new_object



    image = Image.open(mode='r',fp=r'./new_baby.png')
    plt.ion()
    new_image = image.transform(
        method=Image.QUAD,
        data=(0,0,1000,0,1000,1000,0,1000),
        size=(1000,1000)
        )
    # QUAD is 4 points to 4 points transformation. the defalut is to insert the new 'crop' to init at (0,0)
    # returning a new image of a given size (`PIL.Image`)
    # again we will present with `pyplotlib`
    plt.imshow(new_image)
    return new_image

def generate_point_correspondence():
    src_points = np.array([(141, 131), (480, 159), (493, 630),(64, 601)])
    dst_points = np.array([(318, 256),(534, 372),(316, 670),(73, 473)])
    return src_points,dst_points


def create_the_matrix(src_points:np.array,dst_points:np.array) -> np.array:
    """
    Creates the matrix A for which we will have
    Ah = 0 for h=(h₁₁....h₃₃)
    Args
    -----
        `src_points` - list of points from the first image
        `dst_points` - list of points (tuple of integers - pixel location) from the second image, correspoding the first list

    """
    A = []
    assert len(src_points) == len(dst_points),"The lists should be a match of pairs - and hold the sampe length"
    number_of_pairs = len(src_points)
    print(type(A))
    # TODO change this to some numpy trick to build the matrix. maybe with permuatation of the variables within the H vector
    for i in range(0,number_of_pairs):
        x, y = src_points[i] # split the tuple
        a, b = dst_points[i]
        
        # create A_i according to the eq. in the book
        # here we are assuming w_i is one
        A.append([x, y, 1, 0, 0, 0, -a*x, -a*y, -a])
        A.append([0, 0, 0, -x, -y, -1, b*x, b*y, b])
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








