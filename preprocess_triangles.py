import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
import os


def transform_images(load_path, save_path):
    i = 0
    for filename in os.listdir(load_path):
        i = i + 1 
        img = mpimg.imread(os.path.join(load_path, filename))
        img_trans = preprocess_image(preprocess_image(img[4:196,4:196]))
        scipy.misc.imsave(os.path.join(save_path, filename), img_trans)

# one gets an image of size 200*200, reduce it to 100*100. Reduces any image by size 2*2(max pooling)
def preprocess_image(image):
    length = len(image)
    reduced_im = np.zeros((int(length/2),int(length/2)))
    for i in range(0, length, 2):
            for j in range(0, length, 2):
                reduced_im[int(i/2), int(j/2)] = max(image[i,j], image[i+1, j], image[i+1, j+1], image[i, j+1])
    return reduced_im

if __name__ == '__main__':
    load_path = '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/triangle'
    save_path = '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/newtriangle'
    transform_images(load_path, save_path) 

