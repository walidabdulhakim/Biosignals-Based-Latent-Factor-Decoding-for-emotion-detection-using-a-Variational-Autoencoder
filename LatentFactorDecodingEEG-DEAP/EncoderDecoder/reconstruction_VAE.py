from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, LeakyReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mean_squared_error
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import scipy.io as sio

def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    
    batch = K.shape(z_mean)[0]  # Get the batch size
    dim = K.int_shape(z_mean)[1]  # Get the dimension of the latent space
    
    za = K.sqrt(z_log_var)  # Compute the square root of the log variance
    
    epsilon = K.random_normal(shape=(batch, dim))  # Generate random samples from a standard normal distribution
    
    # Reparameterization trick: sample z using the mean and standard deviation
    z = z_mean + (K.sqrt(z_log_var) * epsilon)
    
    return z


 
batch_size = 1024
original_dim = 32
epochs = 10
subNum = 32
zscore = True

# Iterate over subjects
for subNo in range(1, 33):
    # Iterate over latent dimensions
    for latent_dim in range(1, 17):
        print('subNo: ' + str(subNo) + ' latend_dim: ' + str(latent_dim))
        
        # Check if z-score normalization is used
        if zscore:
            sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_zscore\\sub' + str(subNo) + '.mat', 'r')
            x_train = sub_data_file['zscore_data']
        else:
            sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_minmax\\sub' + str(subNo) + '.mat', 'r')
            x_train = sub_data_file['minmax_data']

        x_train = np.transpose(x_train)[:, 0:32]
        x_test = x_train
    
        # Define the encoder network
        inputs = Input(shape=(original_dim, ), name='encoder_input')
        h = Dense(128, activation='relu')(inputs)
        h = Dense(64, activation='relu')(h)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Define the decoder network
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        h_decoded = Dense(64, activation='relu')(latent_inputs)
        h_decoded = Dense(128, activation='relu')(h_decoded)
        inputs_decoded = Dense(original_dim)(h_decoded)
        decoder = Model(latent_inputs, inputs_decoded, name='decoder')
        
        # Connect the encoder and decoder to create the VAE model
        outputs = decoder(z)
        vae = Model(inputs, outputs, name='vae_mlp')
        
        # Define the reconstruction loss and KL divergence loss
        reconstruction_loss = mean_squared_error(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        # Compile the VAE model
        vae.compile(optimizer='adam')
        
        # Train the VAE model
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        
        # Encode the EEG data using the VAE encoder
        encoded_x_zmean = encoder.predict(x_train)[0]
        
        # Save the encoded EEGs to a .mat file
        sio.savemat('D:\\VAE Experiment\\DEAP\\encoded_eegs_vae\\encoded_eegs_vae_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
                    {'encoded_eegs': encoded_x_zmean})
        
        # Uncomment the following lines to save the encoded EEGs with the sampled z values
        # encoded_x_z = encoder.predict(x_train)[2]
        # sio.savemat('D:\\VAE Experiment\\DEAP\\encoded_eegs_vae\\encoded_eegs_vae_z_sub' +
        #            str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
        #            {'encoded_eegs_z': encoded_x_z})
        
        # Uncomment the following lines to save the decoded EEGs
        # decoded_x = vae.predict(x_train)
        # sio.savemat('D:\\VAE Experiment\\DEAP\\decoded_eegs_vae\\decoded_eegs_vae_sub' +
        #            str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
        #            {'decoded_eegs': decoded_x})
