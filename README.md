# "Constraint based circuit design using Generative Adversarial Networks"
This project contains the code used by me in my masters project. 

## Requirements 
keras==2.2.5  
matplotlib  
numpy  
scipy  
pandas  
Pillow  
xlsxwriter  

Along with the above dependencies, the package on Peregrine(HPC clucster) that was used was "Using TensorFlow/1.10.1-fosscuda-2018a-Python-3.6.4". 

## Class Descriptions

**simple_circuits_building :** This class is used to generate the first 2 versions of the circuits dataset. The obstacles are rectangular and only the rows corresponding 
to each of the obstacles are varied within each version of the dataset. The first dataset consists of images of size 48*48 and the second 64*64. The later version 
also has a larger number of obstacles. 

**circuits_building_varied_obstacles :** This class is used to generate the varied obstacles dataset i.e. the final dataset with obstacles of different shapes, sizes and varied positions. In both the circuit building classed one can vary the parameters 'prob_rand_global' and 'prob_rand_local' in the main function so as to have different proportion of global random placement of a wire pixel, local rando placement of a wire pixel and the reward learning based iteration that extends the current wire.  

**non_saturating_GAN :** The implementation of the final architecture that was tested for the non-saturating GAN, is contained here.  In the function train GAN, the 
g_loss variable can be siply be changed to turn the non-saturating GAN to the minimax GAN. The loss for the minimax GAN is commented out and can be changed 
when one needs to run the minimax GAN. the other parameters like batch_size, n_epochs= 200, discrimnator and generator leanring rate can be set in the init function. 

**rms_gan :** The implementation of the final architecture that was tested for the RMS GAN, is contained here. The parameters can be varied in the way discussed above. 

**wgan :** The implementation for the WGAN is showed here. In the function train_GAN, a variable 'd_over_g' is set to 5. It can be varied and it represents the number of
times the discriminator network is trained for each iteration of the generator network. 

**AE_model :** This class contains the implementation of the autoencoder model.  The class also contains the function 'capacity_latent_space' that can be used to test the
capacity of the network by feeding in random latent vectors to the pretrained decoder model. 

**pix2pix_GAN :** This class contains the implementation of the Pix2Pix GAN model. 

**invert :** Loads images from a folder. Flips them and adds the flipped version in the same folder. This was used to increase the size of the dataset, 2 folds.

**preprocess_triangles :** Loads the traingles dataset, and recudes the size of each image from 200*200 to 48*48 with max pooling(2*2). 

The paths from where to load and save, need to be changed in the init function of almost all the classes. 
