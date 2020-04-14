import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pandas import DataFrame
import pandas as pd
from PIL import Image

# loads images from the folder. Flips them and saves them in the same folder
def load_flip_images(load_path, save_path):
    i = 0
    for filename in os.listdir(load_path):
        i = i+1
        p = os.path.join(load_path, filename)
        if p == load_path + '/.DS_Store':
                continue
        img = Image.open(p)
        flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(save_path + '/Image_'+str(i)+'.png')
        flipped_image.save(save_path + '/Image_'+str(i+10000)+'.png')


if __name__ == '__main__':    

    base_path = '/Users/swarajdalmia/Desktop/NeuroMorphicComputing/Code'
    load_path = base_path + '/Data/circuitImages/usefulCircuits/withObstacles_withoutNoise'
    save_path = base_path + '/Data/circuitImages/usefulCircuits/biggerDataset'
    load_flip_images(load_path, save_path)

    # plt.imshow(images[0])
    # plt.show()