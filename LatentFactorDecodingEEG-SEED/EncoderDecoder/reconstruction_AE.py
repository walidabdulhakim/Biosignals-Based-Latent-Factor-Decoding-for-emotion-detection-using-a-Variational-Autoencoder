import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
import scipy.io as sio
import h5py


batch_size = 500
original_dim = 62
epochs = 5
zscore = True

for subNo in range(1,16): #subNo 从1到32
    for latent_dim in range(1,17):
        print('subNo: '+str(subNo)+' latend_dim: '+str(latent_dim))
        # encoder
        eeg_input = Input(shape=(original_dim,))
        dense1 = Dense(256, activation='relu')(eeg_input)
        dense2 = Dense(128, activation='relu')(dense1)
        dense3 = Dense(latent_dim, activation='sigmoid')(dense2)
        encoder = Model(eeg_input, dense3)
        # decoder
        dense4 = Dense(128, activation='relu')(dense3)
        dense5 = Dense(256, activation='relu')(dense4)
        eeg_output = Dense(original_dim)(dense5)

        

        ae = Model(eeg_input, eeg_output)
        ae.compile(optimizer='adam', loss='mean_squared_error')

        #### train the VAE on normalized (z-score) multi-channel EEG data
        if zscore:
            sub_data_file = h5py.File('D:\\Processed SEED\\normalize_zscore\\s'+ str(subNo) +'.mat', 'r')
            x_train = sub_data_file['data']
        else:
            sub_data_file = h5py.File('D:\\Processed DEAP\\normalize_minmax\\sub' + str(subNo) + '.mat', 'r')
            x_train = sub_data_file['minmax_data']

        x_train = np.transpose(x_train)[:, 0:62]
        x_test = x_train

        ae.fit(x_train, x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test))

        # build a model to project inputs
        decoded_x = ae.predict(x_train)
        encoded_x = encoder.predict(x_train)

        
        
        
        sio.savemat('D:\\Processed SEED\\AE\\encoded_eegs_ae\\encoded_eegs_ae_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
                   {'encoded_eegs': encoded_x})

        sio.savemat('D:\\Processed SEED\\AE\\decoded_eegs_ae\\decoded_eegs_ae_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
                    {'decoded_eegs': decoded_x})