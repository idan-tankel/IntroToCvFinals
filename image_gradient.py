# calculate gradient of an BNW image using np
import PIL
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt

def calculate_gradient(img_path):
    image = Image.open(img_path)
    image_as_array = np.array(image)
    
    return np.gradient(image_as_array)
    # gx,gy,*grad = np.gradient(image_as_array)
    # return gx,gy,grad

    

