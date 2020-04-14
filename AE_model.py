import tensorflow as tf
import keras 
from keras.datasets import fashion_mnist
from keras.models import Sequential,Model, model_from_json
from keras.layers import Input, Dense, Activation,Conv2D,MaxPooling2D,Dropout,Flatten,Reshape,UpSampling2D,Conv2DTranspose, AveragePooling2D, BatchNormalization, LeakyReLU
import numpy as np
from keras.callbacks import ModelCheckpoint
from time import time
from keras import optimizers, initializers, losses
from non_saturating_GAN import load_images
import os
import random
import shutil
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt


# import LossHistory

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# the encoder network for triangles. Initializations for kernels ! 
def encode_network(input_img):
    kernel_init = initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed = None)
    bias_init = initializers.Constant(value=0)
    l1 = Conv2D(8, (3,3), strides = 1, padding='same', kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(input_img)  
    l1 = LeakyReLU(alpha = 0.05)(l1)
    l1 = AveragePooling2D((2,2))(l1)   # at this point the output is of shape 8*24*24
    l1 = Conv2D(16,(3,3), strides = 1, padding='same', kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)
    l1 = LeakyReLU(alpha = 0.05)(l1)
    l1 = AveragePooling2D((2,2),padding = 'same')(l1)
    l1 = Conv2D(32,(3,3), strides = 1, padding='same', kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)
    l1 = LeakyReLU(alpha = 0.05)(l1)
    l1 = AveragePooling2D((2,2),padding = 'same')(l1)


    l1 = Flatten()(l1) # at this point the output is of shape 32*16*16 = 8192 and this is reduced to size = 1000
    l1 = Dense(1000, kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)
    l1 = Dense(128, kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)

    return l1

# the decoder network for trinagles
def decode_network(l1, output_bits):
    kernel_init = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
    bias_init = initializers.Constant(value=0.1)


    l1 = Reshape((8,8,2))(l1)

    l1 = UpSampling2D(size=(2,2))(l1)     # nearest neighbour interpolation 
    l1 = Conv2D(64,(3,3), strides = 1, padding='same', kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)
    l1 = BatchNormalization()(l1) 
    l1 = LeakyReLU(alpha = 0.05)(l1)

    l1 = UpSampling2D(size=(2,2))(l1)
    l1 = Conv2D(32,(3,3), strides = 1, padding='same', kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)
    l1 = BatchNormalization()(l1) 
    l1 = LeakyReLU(alpha = 0.05)(l1)
    l1 = UpSampling2D(size=(2,2))(l1)
    l1 = Conv2D(16,(3,3), strides = 1, padding='same', kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)  
    l1 = BatchNormalization()(l1) 
    l1 = LeakyReLU(alpha = 0.05)(l1)

    # 4th transpose of convolutional layer !
    l1 = UpSampling2D(size=(2,2))(l1)
    l2 = Conv2D(output_bits,(3,3), strides = 1, activation='tanh', padding='same', kernel_initializer = kernel_init, bias_initializer = bias_init, use_bias = True)(l1)  

    return l2

# builds different architectures for different image_types(triangles/circuits).
def build_architecture(input_img, output_bits):
    l1 = encode_network(input_img)
    l2 = decode_network(l1, output_bits)   # these signify the color. If col = c, output shape = 128,128,c
    return l2

# input shape needs to be a tensor of dim = (height, width, col), x_train needs to be of shape = _ , height, width, col and image_type = triangle/circuit
def train_AE_model(x_train, input_shape, image_type, save_path, learning_rate_opti= 0.001, num_batch = 40, num_epoch = 100):
    # print("Shape of original training set : ", x_train.shape)
    # input to the AE specified 
    input_img = Input(shape=input_shape)
    l2 = build_architecture(input_img, input_shape[2])
    # The model is specified with the input and the output layers ! 
    model2 = Model(input_img, l2)
    # Specifiyin the optimiser used along with the learning rate
    # opti = optimizers.RMSprop(learning_rate=0.001)
    opti = optimizers.Adam(lr=learning_rate_opti, amsgrad=False)
    # the model is compiled. The optimiser and the loss function is specified ! Label smoothing of 0.1 tried !
    model2.compile(optimizer=opti, loss='mean_squared_error')
    model2.summary()
    # normalise the input images from [0,255] to [-1,1]
    x_train = x_train.astype('float32')
    x_train_1=(x_train-127.5)/127.5
    # this logs the runtime !
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    # adding noise and introducing another dimension !
    # x_train_ = x_train_1 + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)  # see if this adds noise both positive and negative 
    # x_train_ = np.clip(x_train_, -1, 1.)

    # ex = x_train_[3]
    # ex = (ex+1)/2
    # plt.imshow(ex.reshape((128, 128, 4)))
    # plt.show()

    # training the model with inout as noisy images. The loss is calucalted wrt, to the images without noise. Therefore the AE learns how to transform 
    # noise images to those without noise. Also, the epochs and the batch size is specified here !
    history = LossHistory()
    save_checkpoints = ModelCheckpoint(filepath = save_path+'/Epoch_{epoch:02d}-{loss:.6f}.hdf5', monitor='val_loss', verbose = 1, period = 2)

    m = model2.fit(x_train_1, x_train_1, epochs = num_epoch, batch_size = num_batch, callbacks = [save_checkpoints, history]) # set epochs to 150

    # save_losses = np.asarray(history.losses, dtype=np.float32)
    epoch_loss = m.history['loss']

    np.savetxt(save_path+'/loss_history_batchwise.txt', history.losses, fmt='%.6f')
    np.savetxt(save_path+'/loss_history_epochwise.txt', epoch_loss, fmt='%.6f')

    # finds last index of min value in list (we dont -1 cause epochs are saved from 1 instead of 0)
    best_model_index =len(epoch_loss) -  epoch_loss[::-1].index(min(epoch_loss[1::2])) 

    cpy_file = glob.glob(save_path+'/Epoch_'+str(best_model_index)+'*')[0]
    shutil.copyfile(cpy_file,save_path+'/Best_Model_Epoch_'+str(best_model_index)+'.hdf5')

    list_parameters_used = ["Learning Rate = " + str(learning_rate_opti),"Batch Size = " + str(num_batch), "Epochs = "+str(num_epoch)]
    with open(save_path + '/Parameters_used.txt', 'w') as f:
        for item in list_parameters_used:
            f.write("%s\n" % item)

    # iterations = num_epoch*(x_train.shape[0]/num_batch)
    iterations = len(history.losses)
    x = np.linspace(0, iterations, iterations)
    plt.figure()
    plt.plot(x, history.losses, color = 'red')
    plt.ylabel('Loss for the Autoencoder')
    plt.xlabel('Number of batch trainings')
    plt.title('MSE Loss with Batch size = '+str(num_batch)+", Learning Rate = "+str(learning_rate_opti))
    # plt.show()
    # plt.legend('upper right')
    # plt.gca().legend(('Mean_Squared_Error Loss with Adam'))
    plt.savefig(save_path+'/loss.png')
  
    # Saving the model 
    model_json = model2.to_json()
    with open(save_path+'/AE_model_tex.json', "w") as json_file:
        json_file.write(model_json)
    # model2.save_weights(save_path+"/AE_model_"+image_type+version+"_tex.h5")

    # index = 0
    # # this prints image predicted by the AE 
    # pp = model2.predict(np.reshape(x_[index],(1,28,28,1)))
    # pp = np.reshape(pp,(28,28))
    # plt.imshow(pp,cmap='Greys')
    # plt.show()

    # # shows the orinal image without noise
    # plt.imshow(x_train[index],cmap='Greys')


# load path points to the folder where the AE_circuits_are stored. col = 1 for BW and col = 4 for color images. im_size = (height, width)
def load_and_test_AE_model(load_path, test_images, im_size, col):
    # https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
    # loading model
    json_file = open(load_path+'/AE_model_tex.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print("Loaded model from disk")

    # loading different weights for models of each epoch
    im_list = []
    for filename in os.listdir(load_path):
        p = os.path.join(load_path, filename)
        if filename.startswith("Epoch"):
            loaded_model.load_weights(p)
            to_test = random.choice(test_images)
            to_test = to_test[np.newaxis,:,:,:] 
            im_normalised = to_test.astype('float32')
            im_normalised=(im_normalised-127.5)/127.5
            # noisy_input = im_normalised + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=im_normalised.shape)  # see if this adds noise both positive and negative 
            # noisy_input = np.clip(noisy_input, -1, 1.)
            # generated_image = loaded_model.predict(noisy_input, verbose=1)
            generated_image = loaded_model.predict(im_normalised, verbose=1)
            generated_image = (generated_image+1)/2
            # noisy_input = (noisy_input +1)/2

            mpl.use('pdf')

            title_fontsize = 'small'
            fig = plt.figure(dpi=300, tight_layout=True)
            ax = np.zeros(2, dtype=object)
            gs = fig.add_gridspec(1,2)
            ax[0] = fig.add_subplot(gs[0, 0])
            ax[1] = fig.add_subplot(gs[0, 1])
            # ax[2] = fig.add_subplot(gs[1, :])
            if(col == 1):
                ax[0].imshow(np.reshape(to_test,im_size), cmap='gray')
            else: 
                ax[0].imshow(np.reshape(to_test,(image_size[0], image_size[1],col)))
            ax[0].set_title('Image w/o Noise sent as Input to AE', fontsize = title_fontsize)
            ax[0].set_xlabel('(a)')

            # if(col == 1):
            #     ax[1].imshow(np.reshape(noisy_input,im_size), cmap='gray')
            # else: 
            #     ax[1].imshow(np.reshape(noisy_input,(image_size[0], image_size[1],col)))
            # ax[1].set_title("Noisy Image sent as Input to AE", fontsize = title_fontsize)
            # ax[1].set_xlabel('(b)')

            if(col == 1):
                ax[1].imshow(np.reshape(generated_image,im_size), cmap='gray')
            else: 
                ax[1].imshow(np.reshape(generated_image,(image_size[0], image_size[1],col)))
            ax[1].set_title('Image Generated by AE', fontsize = title_fontsize)
            ax[1].set_xlabel('(c)')

            for a in ax:
                a.set_xticks([])
                a.set_yticks([])

            plt.savefig(p[0:-5]+".pdf")
            os.remove(p)

        # for i in range(2):

        #     im = generated_images[i]
        #     # example_1 = np.around(example_1).astype(int)
        #     im = np.reshape(im, (128, 128))
        #     plt.figure()
        #     plt.imshow(im, cmap='gray')
        #     plt.savefig(save_images_to+'/Image_'+str(i))

def capacity_latent_space(load_path,save_path):
    json_file = open(load_path+'/AE_model_tex.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    p = load_path + '/Best_Model_Epoch_148.hdf5'
    loaded_model.load_weights(p)

    encoded_input = Input(shape=(128,))
    # access the start of the autoencoder layers 
    decoder_layer1 = loaded_model.layers[13](encoded_input)
    decoder_layer2 = loaded_model.layers[14](decoder_layer1)
    decoder_layer3 = loaded_model.layers[15](decoder_layer2)
    decoder_layer4 = loaded_model.layers[16](decoder_layer3)
    decoder_layer5 = loaded_model.layers[17](decoder_layer4)
    decoder_layer6 = loaded_model.layers[18](decoder_layer5)
    decoder_layer7 = loaded_model.layers[19](decoder_layer6)
    decoder_layer8 = loaded_model.layers[20](decoder_layer7)
    decoder_layer9 = loaded_model.layers[21](decoder_layer8)
    decoder_layer10 = loaded_model.layers[22](decoder_layer9)
    decoder_layer11 = loaded_model.layers[23](decoder_layer10)
    decoder_layer12 = loaded_model.layers[24](decoder_layer11)
    decoder_layer13 = loaded_model.layers[25](decoder_layer12)
    decoder_layer14 = loaded_model.layers[26](decoder_layer13)
    decoder_layer15 = loaded_model.layers[27](decoder_layer14)
    intermediate_layer_model = Model(encoded_input, decoder_layer15)

    for i in range(1000):
        # +200 to -200 - range of the latent vector space 
        latent_vector = np.random.uniform(-200,200,128)
        latent_vector = latent_vector[np.newaxis,:]
        generated_image = intermediate_layer_model.predict(latent_vector)
        generated_image = (generated_image+1)/2
        plt.imshow(np.reshape(generated_image,(image_size[0], image_size[1],4)))
        plt.savefig(save_path+'/Image_'+str(i), bbox_inches='tight')
        print(i)
        # plt.show()


def run_train(images, load_path, save_path, im_size, im_type, version, learning_rate, batch_size, epoch_size = 0):
    save_path = save_path+im_type+str(version)
    # the input images need to be of a specifid given size !
    # train_AE_model(images[0:200], im_size, im_type, save_path, learning_rate_opti= learning_rate, num_batch = 25, num_epoch = 2)
    train_AE_model(images, im_size, im_type, save_path, learning_rate_opti= learning_rate, num_batch = batch_size, num_epoch = epoch_size)

if __name__ == '__main__':    
    # base_path = '/home/s3494950/thesis'
    base_path = '/Users/swarajdalmia/Desktop/NeuroMorphicComputing/Code'

    # load_path = base_path + '/Data/newtriangle'
    load_path = base_path+'/Data/circuitImages/usefulCircuits/smallerset_obstacles'
    # load_path = base_path+'/Data/withObstacles_withoutNoise'

    col = 4   # set to 4 for color images and 1 for black and white images 
    image_tp = 'circuit'

    ver = 1

    image_size = (128,128)
    save_path = base_path + '/Results/Trained_final_GANs/AE_models/' + image_tp + str(ver)

    images = load_images(load_path, image_size, col)
    images = np.reshape(images, (images.shape[0], image_size[0], image_size[1], col))

    # # plt.imshow(images[0])
    # # plt.show()


    run_train(images, load_path, save_path, im_size = (image_size[0], image_size[1],col), im_type= image_tp, version = ver, learning_rate = 0.001, batch_size = 40, epoch_size= 150)
    load_and_test_AE_model(save_path+image_tp+str(ver), images, image_size, col)
    # run_train(images, load_path, save_path, im_size = (image_size[0], image_size[1],col), im_type= image_tp, version = 2, learning_rate = 0.001, batch_size = 25, epoch_size= 200)
    # load_and_test_AE_model(save_path+image_tp+'2', images, image_size, col)
    # run_train(images, load_path, save_path, im_size = (image_size[0], image_size[1],col), im_type= image_tp, version = 3, learning_rate = 0.01, batch_size = 40, epoch_size= 200)
    # load_and_test_AE_model(save_path+image_tp+'3', images, image_size, col)
    # run_train(images, load_path, save_path, im_size = (image_size[0], image_size[1],col), im_type= image_tp, version = 4, learning_rate = 0.0001, batch_size = 40, epoch_size= 200)
    # load_and_test_AE_model(save_path+image_tp+'4', images, image_size, col)


    # plt.imshow(np.reshape(x_train[0],(28,28)),cmap='Greys')
    # plt.show()
    # print(y_train[0:100])

    # to check sparcity 
    p = base_path + '/Results/Trained_final_GANs/AE_models/check_sparcity'
    capacity_latent_space(save_path, p)
