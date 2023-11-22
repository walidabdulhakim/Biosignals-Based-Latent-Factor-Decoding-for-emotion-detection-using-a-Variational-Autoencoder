% Add the path to the DeeBNet library
addpath('DeeBNet');

subNum = 32;
epochs = 10;

% Iterate over subjects
for subNo = 21:subNum
    % Iterate over latent dimensions
    for latent_dim = 1:16
        % Load z-scored EEG data for the current subject
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub', num2str(subNo), '.mat'));
        zscore_eegs = zscore_eegs_file.zscore_data(:, 1:32);
        
        % Create a data store object to hold the EEG data
        data = DataClasses.DataStore();
        data.valueType = ValueType.gaussian;
        data.trainData = zscore_eegs;
        data.testData = zscore_eegs;
        data.validationData = zscore_eegs;
        
        % Create a Deep Belief Network (DBN) with an autoencoder structure
        dbn = DBN('autoEncoder');
        
        % Configure the parameters for the Restricted Boltzmann Machine (RBM)
        rbmParams = RbmParameters(latent_dim, ValueType.gaussian);
        rbmParams.maxEpoch = epochs;
        rbmParams.gpu = 1;
        rbmParams.samplingMethodType = SamplingClasses.SamplingMethodType.CD;
        rbmParams.performanceMethod = 'reconstruction';
        
        % Add the RBM to the DBN
        dbn.addRBM(rbmParams);
        
        % Train the DBN using the data
        dbn.train(data);
        
        % Perform backpropagation on the DBN
        useGPU = 'yes';
        dbn.backpropagation(data, useGPU);
        
        % Encode the EEGs using the DBN
        encoded_eegs = dbn.getFeature(zscore_eegs);
        
        % Decode the EEGs using the DBN
        decoded_eegs = dbn.reconstructData(zscore_eegs);
        
        % Save the encoded EEGs to a file
        fileName1 = strcat('D:\VAE Experiment\encoded_eegs_1rbm\encoded_eegs_1rbm_sub', num2str(subNo), '_latentdim', num2str(latent_dim));
        save(fileName1, 'encoded_eegs', '-v7.3');
        
        % Save the decoded EEGs to a file
        fileName2 = strcat('D:\VAE Experiment\decoded_eegs_1rbm\decoded_eegs_1rbm_sub', num2str(subNo), '_latentdim', num2str(latent_dim));
        save(fileName2, 'decoded_eegs', '-v7.3');
        
        % Display completion message
        disp(strcat('ends! subject ', num2str(subNo)));
    end
end
