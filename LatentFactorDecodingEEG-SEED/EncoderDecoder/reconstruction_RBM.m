%Test getFeature function for autoEncoder DBN in MNIST data set
% clc
% clear;
addpath('DeeBNet'); 
%
% This code needs the DeeBNet Toolbox V3.0 for runing
%

subNum=15;
%batch_size = 500;
%original_dim = 32;
epochs = 10;

for subNo=1:subNum
    for latent_dim = 1:16
        zscore_eegs_file = load(strcat('D:\Processed SEED\normalize_zscore\s',num2str(subNo),'.mat'));
        zscore_eegs = zscore_eegs_file.data(:,1:62);
        data = DataClasses.DataStore();
        data.valueType = ValueType.gaussian;
        data.trainData = zscore_eegs;
        data.testData = zscore_eegs;
        data.validationData=zscore_eegs;

        dbn=DBN('autoEncoder');
        %dbn.dbnType='autoEncoder';
        % RBM1
    %     rbmParams=RbmParameters(inter_dims(subNo),ValueType.gaussian);
    %     rbmParams.maxEpoch=epochs;
    %     rbmParams.gpu=1;
    %     rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
    %     rbmParams.performanceMethod='reconstruction';
    %     dbn.addRBM(rbmParams);
        % RBM2
        rbmParams=RbmParameters(latent_dim,ValueType.gaussian);
        rbmParams.maxEpoch=epochs;
        rbmParams.gpu=1;
        rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
        rbmParams.performanceMethod='reconstruction';
        dbn.addRBM(rbmParams);

        dbn.train(data);
        useGPU='yes';
        dbn.backpropagation(data,useGPU);

        encoded_eegs=dbn.getFeature(zscore_eegs);
        decoded_eegs=dbn.reconstructData(zscore_eegs);

        %save data
        fileName1 = strcat('D:\Processed SEED\RBM\encoded_eegs_rbm\encoded_eegs_1rbm_sub',num2str(subNo),'_latentdim',num2str(latent_dim));
        save(fileName1,'encoded_eegs','-v7.3');
        fileName2 = strcat('D:\Processed SEED\RBM\decoded_eegs_rbm\decoded_eegs_1rbm_sub',num2str(subNo),'_latentdim',num2str(latent_dim));
        save(fileName2,'decoded_eegs','-v7.3');
        disp(strcat('ends!subject ',num2str(subNo)));
    end
end

