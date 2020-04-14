import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pandas import DataFrame
import pandas as pd
from PIL import Image

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
        #     a = image_list[3]
        #     print(a.shape)      # size of each image is (48, 48)
        #     plt.imshow(a.reshape((im_size[0], im_size[1], col)))
        #     plt.show()
        #     print(1)
    return image_list

def conv2d(x, W):
        # stride is set to 1 and padding=same, adds 0s on either sides
        # fliter : A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels], channels specify color 
        # input : [batch, in_height, in_width, in_channels]
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_2x2(x):
        # horizontal and vertical strides = 4, strides along batch and channel = 1 
        # ksize - the size of window for each dimension of the av_pool operation 
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def up_sampling_2x2(x):
        # use nearest neighbour method to resize 
  return tf.image.resize_images(x, size=(2*x.shape[1], 2*x.shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

def discriminator(x_image, c_dim, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()
        # First Conv and Pool Layers
        W_conv1 = tf.get_variable('d_wconv1', [3, 3, c_dim, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))    # 3rd para set to 3 for color images 
        b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
        h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1) + b_conv1, alpha = 0.05)
        h_pool1 = avg_pool_2x2(h_conv1)

        # Second Conv and Pool Layers
        W_conv2 = tf.get_variable('d_wconv2', [3, 3, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
        h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2, alpha = 0.05)
        h_pool2 = avg_pool_2x2(h_conv2)

        # Third Conv and Pool Layers
        W_conv3 = tf.get_variable('d_wconv3', [3, 3, 16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv3 = tf.get_variable('d_bconv3', [32], initializer=tf.constant_initializer(0))
        h_conv3 = tf.nn.leaky_relu(conv2d(h_pool2, W_conv3) + b_conv3, alpha = 0.05)
        h_pool3 = avg_pool_2x2(h_conv3)

        # First Fully Connected Layer
        # image size is 128*128, and each pooling layer reduces size of output by 2*2. and since there are 3 av_pooling layers.
        # output is reduced by 8*8
        W_fc1 = tf.get_variable('d_wfc1', [16 * 16 * 32, 1000], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable('d_bfc1', [1000], initializer=tf.constant_initializer(0))
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*32])
        h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, alpha= 0.05)

        # Second Fully Connected Layer 
        W_fc2 = tf.get_variable('d_wfc2', [1000, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable('d_bfc2', [128], initializer=tf.constant_initializer(0))
        h_fc2 = tf.nn.leaky_relu(tf.matmul(h_fc1, W_fc2) + b_fc2, alpha= 0.05)
        
        # Third fully connected layer
        W_fc3 = tf.get_variable('d_wfc3', [128, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc3 = tf.get_variable('d_bfc3', [1], initializer=tf.constant_initializer(0))
        y_conv = (tf.matmul(h_fc2, W_fc3) + b_fc3)

        # outputs a value, but this could be any real number between 0,1 so as to minimize loss !
        # out = tf.nn.sigmoid(y_conv, name ='sigmoid') 
        # out = tf.nn.softmax(y_conv)
        out = y_conv   # out is already between 0 and 1. How ?
    return out

def generator(z, batch_size, z_dim, c_dim, reuse=False): 
    with tf.variable_scope('generator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        # reshapring the noise vector of 128 = 8*8*2
        h0 = tf.reshape(z, [batch_size, 8, 8, 2])

        # First up sampling layer that converts from 8*8*2 to 16*16*2
        H_upsampl1 = up_sampling_2x2(h0)
        # First Conv that converts 16*16*2 to 16*16*64
        W_conv1 = tf.get_variable('g_wconv1', [3, 3, 2, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))    # 3rd para set to 3 for color images 
        b_conv1 = tf.get_variable('g_bconv1', [64], initializer=tf.constant_initializer(0))
        H_conv1 = conv2d(H_upsampl1, W_conv1) + b_conv1
        H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
        H_conv1 = tf.nn.leaky_relu(H_conv1, alpha=0.05)

        # Second up sampling layer that converts from 16*16*64 to 32*32*64
        H_upsampl2 = up_sampling_2x2(H_conv1)
        # Second Conv that converts 32*32*64 to 32*32*32
        W_conv2 = tf.get_variable('g_wconv2', [3, 3, 64, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))    # 3rd para set to 3 for color images 
        b_conv2 = tf.get_variable('g_bconv2', [32], initializer=tf.constant_initializer(0))
        H_conv2 = conv2d(H_upsampl2, W_conv2) + b_conv2
        H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
        H_conv2 = tf.nn.leaky_relu(H_conv2, alpha=0.05)

        # third up sampling layer that converts from 32*32*32 to 64*64*32
        H_upsampl3 = up_sampling_2x2(H_conv2)
        # third Conv that converts 64*64*32 to 64*64*16
        W_conv3 = tf.get_variable('g_wconv3', [3, 3, 32, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))    # 3rd para set to 3 for color images 
        b_conv3 = tf.get_variable('g_bconv3', [16], initializer=tf.constant_initializer(0))
        H_conv3 = conv2d(H_upsampl3, W_conv3) + b_conv3
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.leaky_relu(H_conv3, alpha=0.05)

        # fourth up sampling layer that converts from 64*64*16 to 128*128*16
        H_upsampl4 = up_sampling_2x2(H_conv3)
        # fourth conv layers converts from 128*128*16 to 128*128 * col
        W_conv4 = tf.get_variable('g_wconv4', [3, 3, 16, c_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))    # 3rd para set to 3 for color images 
        b_conv4 = tf.get_variable('g_bconv4', [c_dim], initializer=tf.constant_initializer(0))
        H_conv4 = conv2d(H_upsampl4, W_conv4) + b_conv4
        out = tf.nn.tanh(H_conv4)
    return out

def test_generator_run():
    sess = tf.Session()
    z_dimensions = 100
    # place holder variable that can be filled during runtime. Here, batch size can be varied 
    # since we put in None
    z_test_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])
    # a variable in the tensor map
    sample_image = generator(z_test_placeholder, 1, z_dimensions)       
    # the input variable we will use 
    test_z = np.random.uniform(-1, 1, [1,z_dimensions])
    sess.run(tf.global_variables_initializer())
    temp = (sess.run(sample_image, feed_dict={z_test_placeholder: test_z}))
    my_i = temp.squeeze()
    plt.imshow(my_i, cmap='gray_r')
    plt.show()

def train_GAN(images, save_path, ckpt_save_name, batch_size, n_epochs, dsrmntr_l_rate,  gnrtr_l_rate , load = False, checkpoint_load = "Null"):
        # the hyperparameters that need to be varied/chnaged !
        z_dimensions = 128   # need to change architecture if this needs to be varied 
        color = images.shape[3]
        hyper_para = ['z_dimensions', 'batch_size', 'discriminator_l_rate', 'generator_l_rate', 'num_epochs']
        hyper_para_values = [z_dimensions, batch_size, dsrmntr_l_rate, gnrtr_l_rate, n_epochs]

        x_placeholder = tf.placeholder("float", shape = [None,images.shape[1],images.shape[2],color]) #Placeholder for input images to the discriminator
        z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions]) #Placeholder for input noise vectors to the generator
        Dx = discriminator(x_placeholder, color) #Dx will hold discriminator prediction probabilities for the real MNIST images
        Gz = generator(z_placeholder, batch_size, z_dimensions, color) #Gz holds the generated images
        Dg = discriminator(Gz, color, reuse=True) #Dg will hold discriminator prediction probabilities for generated images

        # RMS loss GAN
        # discriminator: minimize (D(x) – 1)^2 + (D(G(z)))^2
        d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(labels= Dx, predictions = tf.ones_like(Dg)))
        d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(labels= Dg, predictions = tf.zeros_like(Dg)))
        d_loss = d_loss_real + d_loss_fake 

        # generator: minimize (D(G(z)) – 1)^2
        g_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels= Dg, predictions = tf.ones_like(Dg)))


        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]

        # Specify a learning rate if need be. Default is 0.001 
        adamD = tf.train.AdamOptimizer(learning_rate =  dsrmntr_l_rate)
        adamG = tf.train.AdamOptimizer(learning_rate = gnrtr_l_rate)
        trainerD = adamD.minimize(d_loss, var_list=d_vars)
        trainerG = adamG.minimize(g_loss, var_list=g_vars)

        sample_image = generator(z_placeholder, 1, z_dimensions, c_dim=color, reuse=True)   # change 1 to how many ever images you might want 

        init_op = tf.global_variables_initializer()
        # uptill now we have defined the TF graph. Now we crate the saver at the end of our graph
        saver = tf.train.Saver()
        # now to start the training we create a session and start. 
        sess = tf.Session()
        if(load == True):
                #First let's load meta graph and restore weights
                # saver = tf.train.import_meta_graph((load_path + '.meta'))
                # saver.restore(sess, tf.train.latest_checkpoint('./'))
                saver.restore(sess, save_path+"/"+checkpoint_load)
                x = 5/0 # IMPORTANT : save the graphs differently and also number the images differently_otherwise there will be an issue 
        else: 
                sess.run(init_op)

        discriminator_loss = []
        generator_loss = [] 
        discriminator_loss_real = []
        discriminator_loss_fake = []
        accuracy_fake = []
        accuracy_real = []

        # normalise the images
        im_normal = images.astype('float32')
        im_normal=(im_normal-127.5)/127.5

        batch_iterations = int((n_epochs * images.shape[0])/batch_size)
        for i in range(batch_iterations):
                z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dimensions])
                real_image_batch = im_normal[np.random.randint(im_normal.shape[0], size = batch_size), :,:,:]
                _,dLoss,dLossFake, dLossReal, realProb, fakeProb = sess.run([trainerD, d_loss, d_loss_fake, d_loss_real, Dx, Dg],feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch}) #Update the discriminator
                _,gLoss = sess.run([trainerG,g_loss],feed_dict={z_placeholder:z_batch})
                discriminator_loss.append(dLoss)
                generator_loss.append(gLoss)
                discriminator_loss_fake.append(dLossFake)
                discriminator_loss_real.append(dLossReal)
                realProb = [x - 0.5 for x in realProb]
                fakeProb = [x - 0.5 for x in fakeProb]
                acc_real = sum(x > 0 for x in realProb)
                acc_fake = sum(x < 0 for x in fakeProb)
                accuracy_fake.append(acc_fake/batch_size)
                accuracy_real.append(acc_real/batch_size)
                if (i % int(im_normal.shape[0]/batch_size) == 0 and i > 1):
                        z_batch = np.random.uniform(-1, 1, size=[1, z_dimensions])
                        temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
                        my_i = temp.squeeze() # removes the dimensions whose size is 1 in the tensor 
                        my_i = (my_i+1)/2
                        plt.figure()
                        plt.imshow(my_i)
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig(save_path+'/Epoch_'+str(int(i/int(im_normal.shape[0]/batch_size)))+'.pdf')

                # if(i % 500 == 0):
                #         x = np.linspace(0, i+1, i+1)
                #         plt.figure()
                #         plt.plot(x, discriminator_loss, color = 'blue')
                #         plt.plot(x, generator_loss, color = 'orange')
                #         plt.ylabel('Loss')
                #         plt.xlabel('Number of iterations')
                #         # plt.show()
                #         # plt.legend('upper right')
                #         plt.title("Loss after "+str(i)+" iterations")
                #         plt.gca().legend(('discriminator','generator'))
                #         plt.savefig(save_path+'/loss.png')
        saver.save(sess, save_path+"/"+ckpt_save_name)
        save_plots_and_paras(save_path, batch_iterations, discriminator_loss, generator_loss, discriminator_loss_fake, discriminator_loss_real, accuracy_fake,
        accuracy_real, hyper_para, hyper_para_values)

def save_plots_and_paras(save_path, batch_iterations, discriminator_loss, generator_loss, discriminator_loss_fake, discriminator_loss_real, accuracy_fake,
accuracy_real, hyper_para, hyper_para_values):
        x = np.linspace(0, batch_iterations, batch_iterations)
        plt.figure()
        plt.plot(x, discriminator_loss, color = 'blue')
        plt.plot(x, generator_loss, color = 'orange')
        plt.ylabel('Loss')
        plt.xlabel('Number of iterations')
        # plt.show()
        # plt.legend('upper right')
        plt.gca().legend(('discriminator','generator'))
        plt.savefig(save_path+'/loss.pdf')

        plt.figure()
        plt.plot(x, discriminator_loss_fake, color = 'red')
        plt.plot(x, discriminator_loss_real, color = 'green')
        plt.ylabel('Loss')
        plt.xlabel('Number of iterations')
        # plt.show()
        # plt.legend('upper right')
        plt.gca().legend(('discriminator loss for fake images','discriminator loss for real images'))
        plt.savefig(save_path+'/loss_real_fake.pdf')

        plt.figure()
        plt.plot(x, accuracy_fake, 'o', color = 'purple')
        plt.plot(x, accuracy_real, 'o', color = 'orange')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of iterations')
        # plt.show()
        # plt.legend('upper right')
        plt.gca().legend(('accuracy for fake images','accuracy for real images'))
        plt.savefig(save_path+'/accuracy.pdf')

        writer = pd.ExcelWriter(save_path+'/loss.xlsx', engine='xlsxwriter')

        df1 = DataFrame({'Generator Loss': generator_loss, 'Discriminator Loss': discriminator_loss, 
        'Discriminator Loss for Real Images': discriminator_loss_real, 'Discriminator Loss for Fake Images': discriminator_loss_fake,
        'Accuracy for Real Images': accuracy_real, 'Accuracy for Fake Images': accuracy_fake})
        df2 = DataFrame({'Hyper Parameters': hyper_para, 'Corresponding Values': hyper_para_values})

        df1.to_excel(writer, sheet_name='sheet1', index=False)
        df2.to_excel(writer, sheet_name='sheet2', index=False)
        writer.save()

if __name__ == '__main__':
        base_path = '/home/s3494950/thesis'
        # base_path = '/Users/swarajdalmia/Desktop/NeuroMorphicComputing/Code'

        # load_path = base_path + '/Data/newtriangle'
        # load_path = base_path+'/Data/circuitImages/usefulCircuits/asIs'
        load_path = base_path+'/Data/biggerDataset'

        col = 4   # set to 4 for color images and 1 for black and white images 
        image_tp = 'circuit'
        ckpt_save_name = "checkpoint_1"
        # ckpt_load = "checkpoint_n"
        # load = False
        image_size = (128,128)
        save_path = base_path + '/Results/rms_gan/' + image_tp
        images = load_images(load_path, image_size, col)
        images = np.reshape(images, (images.shape[0], image_size[0], image_size[1], col))

        ver = 1
        batch_s = 40    # vaired between 40, 80, 120 
        num_ep = 150
        d_lr = 0.0005   # varied between 0.0005, 0.001 and 0.0001
        g_lr = 0.0005 

        # check if its possible to have epochs be divisible by batch size 
        train_GAN(images, save_path+str(ver), ckpt_save_name, batch_size = batch_s, n_epochs= num_ep, dsrmntr_l_rate= d_lr, gnrtr_l_rate=g_lr)
        # train_GAN(images, save_path+str(2), ckpt_save_name, batch_size = 40, n_epochs= 200, dsrmntr_l_rate= 0.005, gnrtr_l_rate=0.001)
        # train_GAN(images, save_path+str(4), ckpt_save_name, batch_size = 25, n_epochs= 200, dsrmntr_l_rate= 0.001, gnrtr_l_rate=0.001)

