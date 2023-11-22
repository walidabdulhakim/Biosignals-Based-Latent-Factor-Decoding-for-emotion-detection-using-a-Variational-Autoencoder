subNum = 32;
channelNum = 32;
zscore = 1;

latdim = 16;

for subNo = 1:subNum
    for latent_dim = 1:latdim
        % Load z-scored EEG data for the current subject
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub', num2str(subNo), '.mat'));
        zscore_eegs = zscore_eegs_file.zscore_data(:, 1:channelNum)';
        
        % Display subject number and latent dimension
        disp(strcat('subNo: ', num2str(subNo), ' latentdim: ', num2str(latent_dim)));
        
        % Perform Independent Component Analysis (ICA)
        [ICs, A, W] = fastica(zscore_eegs, 'numOfIC', latent_dim, 'verbose', 'off');
        
        % Decode the EEGs using ICA components
        decoded_eegs = A * ICs;
        
        % Store the encoded EEGs and save to a file
        encoded_eegs = ICs;
        fileName = strcat('D:\VAE Experiment\DEAP\encoded_eegs_ica\encoded_eegs_ica_sub', num2str(subNo), '_latedtdim', num2str(latent_dim));
        save(fileName, 'encoded_eegs', '-v7.3');
    end
end
