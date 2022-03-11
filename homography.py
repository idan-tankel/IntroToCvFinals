import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# class homography():

def apply_homography(image):
    """
    this function is applying a simple homography on an image
    Args:
        imgae: the PIL image to apply homography on
    """
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




