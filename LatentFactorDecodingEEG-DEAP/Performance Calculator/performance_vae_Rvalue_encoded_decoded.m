subNum = 32;
channelNum = 32;
latdimNum = 16;

corr_chs = zeros(latdimNum, subNum, channelNum);

% Calculate correlation coefficients between VAE decoded EEGs and z-scored EEGs
for latdim = 1:latdimNum
    for subNo = 1:subNum
        % Load the decoded EEGs from VAE
        filename1 = strcat('D:\VAE Experiment\DEAP\decoded_eegs_vae\decoded_eegs_vae_sub', num2str(subNo), '_latentdim', num2str(latdim), '.mat');
        decoded_eegs_ae_file = load(filename1);
        decoded_eegs_ae = decoded_eegs_ae_file.decoded_eegs;
        
        % Load the z-scored EEGs
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub', num2str(subNo), '.mat'));
        zscore_eegs = zscore_eegs_file.zscore_data;
        
        for chno = 1:channelNum
            % Compute correlation coefficient between VAE decoded EEGs and z-scored EEGs for each channel
            R = corrcoef(decoded_eegs_ae(:, chno), zscore_eegs(:, chno));
            corr_chs(latdim, subNo, chno) = R(1, 2);
        end
    end
end

ms = zeros(1, latdimNum);
% Calculate mean correlation for each latent dimension
for latdim = 1:latdimNum
   latdim_performance = squeeze(corr_chs(latdim, :, :));
   m = mean(mean(latdim_performance));
   ms(1, latdim) = m;
end

% Save the calculated correlation coefficients to a .mat file
fileName = 'C:\Users\LEGION\Desktop\LatentFactorDecodingEEG-master\calclulated_reconstruction_performance\vae_reconstruction_corr_Rvalue.mat';
save(fileName, 'corr_chs', '-v7.3');
