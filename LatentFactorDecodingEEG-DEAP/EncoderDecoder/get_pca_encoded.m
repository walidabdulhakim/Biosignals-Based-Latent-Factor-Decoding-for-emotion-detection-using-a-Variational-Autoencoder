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
        
        % Perform Principal Component Analysis (PCA)
        X = zscore_eegs.';
        [coeff, score, latent] = pca(X);
        
        % Encode the EEGs using PCA scores
        encoded_eegs = score(:, 1:latent_dim);
        
        % Store the encoded EEGs and save to a file
        fileName = strcat('D:\VAE Experiment\DEAP\encoded_eegs_pca\encoded_eegs_pca_sub', num2str(subNo), '_latedtdim', num2str(latent_dim));
        save(fileName, 'encoded_eegs', '-v7.3');
    end
end
