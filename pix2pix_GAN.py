from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import os
from pandas import DataFrame
import pandas as pd
from PIL import Image
from random import randint
import copy
import shutil
import glob



# define the discriminator model
def define_discriminator(image_shape, learning_rate_discriminator = 0.0002):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
    # target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=learning_rate_discriminator, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g
 
# define the standalone generator model
def define_generator(image_shape=(128,128,4)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	# e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
	b = Activation('relu')(b)
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	# d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(b, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(4, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape, learning_rate_generator = 0.0002):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input. The input to the generator are 
    # images with only obstacles
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=learning_rate_generator, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model



# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset 
    image_obsta, image_paths_n_obsta = dataset
    # choose random instances
    indices = list(range(0,image_obsta.shape[0]))
    random.shuffle(indices)
    ix = indices[0:n_samples]
	# retrieve selected images
    X1, X2 = image_obsta[ix], image_paths_n_obsta[ix]
	# generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# # extracts path images. Its given a batch of color images with obstacles and paths 
# # implement a thresholding method to extract only the path i.e. remove the obstacles
# def extract_path_image(imgs, im_size = 128):

#     size = imgs.shape[0]
#     print("size is : ", size)
#     for i in range(size):
#         im = imgs[i]
#         for j in range(im_size):
#             for k in range(im_size):
#                 pixel = im[j][k]
#                 # remove the obstacles
#                 if(pixel[1]>80 and pixel[0]<40 and pixel[2]<40):
#                     im[j][k] = [0,0,0,255]
#     return imgs

# input is a set of color images with obstacles and paths. It removed the paths and outputs the set of images with 
# only the obstacles
def remove_paths(imgs, im_size = 128):
    size = imgs.shape[0]
    for i in range(size):
        im = imgs[i]
        for j in range(im_size):
            for k in range(im_size):
                pixel = im[j][k]
                # remove the white paths 
                if((abs((pixel[0]-pixel[1])/2)<10 and abs((pixel[1]-pixel[2])/2)<10) or (pixel[0]>=100 and pixel[1]>=100 and pixel[2]>=100)):
                    im[j][k] = [0,0,0,255]      # convert white pixels to black
                # remove the blue paths 
                elif((pixel[2]>=80 and pixel[0]<pixel[2]-20 and pixel[1]<pixel[2]-20 and pixel[0]<80 and pixel[1]<80) or (pixel[0]<40 and pixel[1]<40 and pixel[2]<80)):
                    im[j][k] = [0,0,0,255]
    return imgs


# train pix2pix models
def train_save(save_path, d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1, n_patch=8):
	# calculate the number of batches per training epoch
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
    generator_loss = [] 
    discriminator_loss = []
    discriminator_loss_real = []
    discriminator_loss_fake = []
    for i in range(n_steps):
        # select a batch of real samples
        [real_image_obsta_batch, real_image_paths_n_obsta_batch], label_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        fake_image_paths_n_obsta, label_fake = generate_fake_samples(g_model, real_image_obsta_batch, n_patch)
        # update discriminator for real samples 
        d_loss1 = d_model.train_on_batch([real_image_obsta_batch, real_image_paths_n_obsta_batch], label_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([real_image_obsta_batch, fake_image_paths_n_obsta], label_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(real_image_obsta_batch, [label_real, real_image_paths_n_obsta_batch])

        #  store the images that the generator generates after each epoch
        if(i % bat_per_epo == 0):
            [real_image_obsta_sample, real_image_paths_n_obsta_sample], label_real = generate_real_samples(dataset, 1, n_patch)
            generated_image = g_model.predict(real_image_obsta_sample)
            mpl.use('pdf')
            title_fontsize = 'small'

            fig = plt.figure(dpi=300, tight_layout=True)
            ax = np.zeros(2, dtype=object)
            gs = fig.add_gridspec(1,2)
            ax[0] = fig.add_subplot(gs[0, 0])
            ax[1] = fig.add_subplot(gs[0, 1])
            ax[0].imshow(np.reshape(real_image_paths_n_obsta_sample,(128, 128, 4)).astype('uint8'))
            ax[0].set_title('Original Image', fontsize = title_fontsize)
            ax[0].set_xlabel('(a)')
            ax[1].imshow(np.reshape(generated_image,(128, 128, 4)))
            ax[1].set_title('Image Generated by Generator', fontsize = title_fontsize)
            ax[1].set_xlabel('(b)')
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            plt.savefig(save_path +'/Epoch_'+ str(int(i/bat_per_epo))+"_paths.pdf")

            fig2 = plt.figure(dpi=300, tight_layout=True)
            ax = np.zeros(2, dtype=object)
            gs = fig2.add_gridspec(1,2)
            ax[0] = fig2.add_subplot(gs[0, 0])
            ax[1] = fig2.add_subplot(gs[0, 1])
            ax[0].imshow(np.reshape(real_image_obsta_sample,(128, 128, 4)).astype('uint8'))
            ax[0].set_title('Original Image', fontsize = title_fontsize)
            ax[0].set_xlabel('(a)')
            ax[1].imshow(np.reshape(generated_image,(128, 128, 4)))
            ax[1].set_title('Image Generated by Generator', fontsize = title_fontsize)
            ax[1].set_xlabel('(b)')
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            plt.savefig(save_path +'/Epoch_'+ str(int(i/bat_per_epo))+"_obst.pdf")

        discriminator_loss_real.append(d_loss1)
        discriminator_loss_fake.append(d_loss2)
        generator_loss.append(g_loss)
        discriminator_loss.append(d_loss1+d_loss2)
        print(i)

    # save the plots for loss etc 
    x = np.linspace(0, n_steps, n_steps)
    plt.figure()
    plt.plot(x, discriminator_loss, color = 'blue')
    plt.ylabel('Discriminator Loss')
    plt.xlabel('Number of iterations')
    # plt.show()
    # plt.legend('upper right')
    # plt.gca().legend(('discriminator','generator'))
    plt.savefig(save_path+'/loss_discriminator.pdf')

    plt.figure()
    plt.plot(x, generator_loss, color = 'orange')
    plt.ylabel('Generator Loss')
    plt.xlabel('Number of iterations')
    # plt.show()
    # plt.legend('upper right')
    # plt.gca().legend(('discriminator loss for fake images','discriminator loss for real images'))
    plt.savefig(save_path+'/loss_generator.pdf')

    writer = pd.ExcelWriter(save_path+'/loss.xlsx', engine='xlsxwriter')
    df1 = DataFrame({'Generator Loss': generator_loss, 'Discriminator Loss': discriminator_loss, 'Discriminator Loss for Real Images': discriminator_loss_real, 'Discriminator Loss for Fake Images': discriminator_loss_fake})
    df1.to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()

    # Saving the Gnerator Model and weights since that is the only one necessary 
    model_json = g_model.to_json()
    with open(save_path+'/Generator_model_tex.json', "w") as json_file:
        json_file.write(model_json)
    g_model.save_weights(save_path+'/Generator_model_weights_tex.h5')


def load_model_and_check(load_path, test_data):
    json_file = open(load_path+'/Generator_model_tex.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print('Model loaded')
    loaded_model.load_weights(load_path+'/Generator_model_weights_tex.h5')

    for i in range(test_data.shape[0]):
        rand_im = test_data[i]
        rand_im = rand_im[np.newaxis,:,:,:] 
        generated_image = loaded_model.predict(rand_im)

        mpl.use('pdf')
        title_fontsize = 'small'
        fig = plt.figure(dpi=300, tight_layout=True)
        ax = np.zeros(2, dtype=object)
        gs = fig.add_gridspec(1,2)
        ax[0] = fig.add_subplot(gs[0, 0])
        ax[1] = fig.add_subplot(gs[0, 1])

        ax[0].imshow(np.reshape(rand_im,(128, 128, 4)).astype('uint8'))
        ax[0].set_title('Test Image as Input', fontsize = title_fontsize)
        ax[0].set_xlabel('(a)')
        ax[1].imshow(np.reshape(generated_image,(128, 128, 4)))
        ax[1].set_title('Image Generated by Generator', fontsize = title_fontsize)
        ax[1].set_xlabel('(b)')
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        plt.savefig(load_path +'/Test_Image_Level4_'+ str(i)+'.pdf')


def load_images(folder, im_size = (128,128), col = 1):
    # load color images after resizing them !
    im_list = []
    for filename in os.listdir(folder):
        p = os.path.join(folder, filename)
        if p == folder + '/.DS_Store':
                continue
        # img = mpimg.imread(p)
        if(col == 1):
                img = Image.open(p).convert('L')
        else:
                img = Image.open(p)
        im_resize = img.resize(im_size, Image.ANTIALIAS)
        im_list.append(np.ravel(im_resize)) # flattened the images, we need to reshape them before printing
    image_list = np.array(im_list)
    return image_list

if __name__ == '__main__':    
    # define image shape
    image_shape = (128,128,4)
    image_size = (128,128)
    col = 4   # set to 4 for color images and 1 for black and white images 
    image_tp = 'circuit'

#-------------------------------

    ver = 13
    lr_discriminator = 0.0001
    lr_generator = 0.001
    num_epochs = 5
    num_batch = 1  # ensure that the batch size dives the number of samples entirely

    # base_path = '/home/s3494950/thesis'
    base_path = '/Users/swarajdalmia/Desktop/NeuroMorphicComputing/Code'

    # load_path = base_path+'/Data/circuitImages/usefulCircuits/withObstacles_withoutNoise'
    load_path = base_path+'/Data/circuitImages/usefulCircuits/smallerset_obstacles'   # 56 items 
    # load_path = base_path+'/Data/biggerDataset'

    save_path = base_path + '/Results/Trained_final_GANs/pix2pix/circuit_' + str(ver)

#-------------------------------

    # images = load_images(load_path, image_size, col)
    # images = np.reshape(images, (images.shape[0], image_size[0], image_size[1], col))

    # d_model = define_discriminator(image_shape, learning_rate_discriminator=lr_discriminator)
    # g_model = define_generator(image_shape)
    # gan_model = define_gan(g_model, d_model, image_shape, learning_rate_generator=lr_generator)
    
    # # load image data. [image_obsta, image_paths_n_obsta]
    # im = copy.deepcopy(images)
    # dataset = [remove_paths(im),images]
    # print("removed paths")
    # # train model
    # train_save(save_path, d_model, g_model, gan_model, dataset, n_epochs = num_epochs, n_batch=num_batch)
    
    p =  '/Users/swarajdalmia/Desktop/NeuroMorphicComputing/Code/Data/circuitImages/usefulCircuits/test_obstacles'
    testing_data = load_images(p, image_size, col)
    testing_data = np.reshape(testing_data, (testing_data.shape[0], image_size[0], image_size[1], col))
    load_model_and_check(save_path, testing_data)




           
