from keras.layers  import Input, LSTM, Dense, Dropout
from keras.models import Model, Sequential
import h5py
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mat73
import os
from keras.optimizers import SGD


trialL=400
trialNum=30
latdim=16
step=400 
method='rbm'
batch=15
epoch_num=5

def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x


l_data = [1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0]
l_data =np.transpose(l_data)
print(l_data.shape)


pred_test_f1=[]
pred_train_f1=[]
pred_test_acc=[]
pred_train_acc=[]

for testSubNo in range(1,16):
    X_train = []  

    print('test subNo: '+str(testSubNo))

    for trainSubNo in range(1,16):
        file2 = mat73.loadmat('D:\\Processed SEED\\' + method + '\\encoded_eegs_' + method + '\\encoded_eegs_1'+ method + '_sub' + str(
                trainSubNo) + '_latentdim' + str(latdim) + '.mat')
        trainSubData = file2['encoded_eegs']
        #trainSubData = np.transpose(trainSubData) # for ica only
        trainSubData = ZscoreNormalization(trainSubData)
        for trialNo in range(0, 30):
             trial_data = trainSubData[trialNo * trialL :(trialNo + 1) * trialL, :]
             trial_data = trial_data[0:trialL:step, :]
             X_train.append(trial_data)

    print(len(X_train))
    
     
    
    seqL = trial_data.shape[0]
    print('seqence length: '+str(seqL))
    model = Sequential()
    model.add(LSTM(1024, input_shape=(seqL, latdim)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    if len(os.listdir(('saved_model/'))) != 0:
        model.load_weights('saved_model/')

    train, test = train_test_split(np.array(X_train), test_size=0.3, random_state=42, shuffle=True)
    ytrain, ytest = train_test_split(l_data, test_size=0.3, random_state=42, shuffle=True)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train,ytrain, validation_data=(test,ytest), epochs=epoch_num, batch_size=batch, shuffle=True)
    #model.save_weights('saved_model/')
    # predict on the training data after training...
    #Y_test_predict = model.predict_classes([X_test])
    #Y_train_predict = model.predict_classes([X_train])    
    
    Y_test_predict = model.predict(np.array(test))
    Y_test_predict = (Y_test_predict > 0.5).astype("int32")

    Y_train_predict = model.predict(np.array(train))
    Y_train_predict = (Y_train_predict > 0.5).astype("int32")

    print(metrics.f1_score(ytest, Y_test_predict))
    print(metrics.accuracy_score(ytest, Y_test_predict))
    print(metrics.f1_score(ytrain, Y_train_predict))
    print(metrics.accuracy_score(ytrain, Y_train_predict))

    pred_test_f1.append(metrics.f1_score(ytest, Y_test_predict))
    pred_test_acc.append(metrics.accuracy_score(ytest, Y_test_predict))
    pred_train_f1.append(metrics.f1_score(ytrain, Y_train_predict))
    pred_train_acc.append(metrics.accuracy_score(ytrain, Y_train_predict))



sio.savemat('lstm_personal_performance_'+ method + '.mat',
        {'test_f1': pred_test_f1, 'test_acc': pred_test_acc, 'train_f1': pred_train_f1,
            'train_acc': pred_train_acc})
