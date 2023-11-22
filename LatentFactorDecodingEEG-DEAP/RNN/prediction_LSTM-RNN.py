from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model, Sequential
import h5py
import scipy.io as sio
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import mat73
import os

# Set constants and parameters
trialL = 63 * 128
trialNum = 40
latdim = 16
step = 128 * 3
method = '1rbm'
emodim = 0  # valence (0) or arousal (1)
batch = 40
epoch_num = 5

def ZscoreNormalization(x):
    """Z-score normalization"""
    x = (x - np.mean(x)) / np.std(x)
    return x

# Load trial labels data
file = h5py.File('C:\\Users\\LEGION\\Desktop\\graduation codes\\LatentFactorDecodingEEG-master\\trial_labels_personal_valence_arousal_dominance.mat', 'r')
l_data = file['trial_labels']
l_data = np.transpose(l_data)
print(l_data.shape)

pred_test_f1 = []
pred_train_f1 = []
pred_test_acc = []
pred_train_acc = []

# Iterate through test subjects
for testSubNo in range(1, 33):
    X_train = []
    print('test subNo: ' + str(testSubNo))

    # Iterate through train subjects
    for trainSubNo in range(1, 33):
        file2 = mat73.loadmat('D:\\VAE Experiment\\DEAP\\encoded_eegs_' + method + '\\encoded_eegs_'+ method + '_sub' + str(trainSubNo) + '_latentdim' + str(latdim) + '.mat')
        trainSubData = file2['encoded_eegs']
        trainSubData = ZscoreNormalization(trainSubData)
        for trialNo in range(0, 40):
            trial_data = trainSubData[trialNo * trialL:(trialNo + 1) * trialL, :]
            trial_data = trial_data[0:trialL:step, :]
            X_train.append(trial_data)

    seqL = trial_data.shape[0]
    print('sequence length: ' + str(seqL))
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(200, input_shape=(seqL, latdim)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Load pre-trained weights if available
    if len(os.listdir(('C:\\Users\\LEGION\\Desktop\\graduation codes\\LatentFactorDecodingEEG-master\\saved_model'+str(emodim)+'/'))) != 0:
        model.load_weights('C:\\Users\\LEGION\\Desktop\\graduation codes\\LatentFactorDecodingEEG-master\\saved_model'+str(emodim)+'/')
    
    # Split the data into training and testing sets
    train, test = train_test_split(np.array(X_train), test_size=0.3, random_state=42, shuffle=True)
    ytrain, ytest = train_test_split(l_data[:, emodim], test_size=0.3, random_state=42, shuffle=True)
    
    # Compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(train, ytrain, validation_data=(test, ytest), epochs=epoch_num, batch_size=batch, shuffle=True)
    
    # Make predictions on test and train data
    Y_test_predict = model.predict(np.array(test))
    Y_test_predict = (Y_test_predict > 0.5).astype("int32")

    Y_train_predict = model.predict(np.array(train))
    Y_train_predict = (Y_train_predict > 0.5).astype("int32")

    # Calculate and print performance metrics
    print(metrics.f1_score(ytest, Y_test_predict))
    print(metrics.accuracy_score(ytest, Y_test_predict))
    print(metrics.f1_score(ytrain, Y_train_predict))
    print(metrics.accuracy_score(ytrain, Y_train_predict))

    # Store the performance metrics
    pred_test_f1.append(metrics.f1_score(ytest, Y_test_predict))
    pred_test_acc.append(metrics.accuracy_score(ytest, Y_test_predict))
    pred_train_f1.append(metrics.f1_score(ytrain, Y_train_predict))
    pred_train_acc.append(metrics.accuracy_score(ytrain, Y_train_predict))

# Save the performance metrics to a .mat file
if emodim == 0:
    sio.savemat('C:\\Users\\LEGION\\Desktop\\graduation codes\\LatentFactorDecodingEEG-master\\lstm_personal_performance_'+method+'_weighted_valence_subcross.mat',
                {'test_f1': pred_test_f1, 'test_acc': pred_test_acc, 'train_f1': pred_train_f1,
                 'train_acc': pred_train_acc})
else:
    sio.savemat('C:\\Users\\LEGION\\Desktop\\graduation codes\\LatentFactorDecodingEEG-master\\lstm_personal_performance_'+method+'_weighted_arousal_subcross.mat',
                {'test_f1': pred_test_f1, 'test_acc': pred_test_acc, 'train_f1': pred_train_f1,
                 'train_acc': pred_train_acc})
