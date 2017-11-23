import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def convert_to_gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

def show_image(bnwimage, stream):
    plt.imshow(bnwimage, cmap = plt.get_cmap(stream))
    plt.show()

def get_image(filename):
    return mpimg.imread(filename)

def save_image(filename, img, stream):
    mpimg.imsave(filename, img,cmap = plt.get_cmap(stream))