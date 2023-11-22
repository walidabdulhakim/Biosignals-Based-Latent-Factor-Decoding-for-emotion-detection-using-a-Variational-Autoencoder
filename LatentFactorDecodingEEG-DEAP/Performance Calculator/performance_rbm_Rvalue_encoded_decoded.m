subNum = 32;
channelNum = 32;
latent_dims = 1:1:16;

corr_chs = zeros(length(latent_dims), subNum, channelNum);

for subNo = 1:subNum
    for j = 1:length(latent_dims)
        disp(strcat('subNo: ', num2str(subNo), ' latentdim: ', num2str(latent_dims(j))));
        
        % Load the decoded EEGs from RBM
        filename1 = strcat('D:\VAE Experiment\DEAP\decoded_eegs_1rbm\decoded_eegs_1rbm_sub', num2str(subNo), '_latentdim', num2str(latent_dims(j)), '.mat');
        decoded_eegs_rbm_file = load(filename1);
        decoded_eegs_rbm = decoded_eegs_rbm_file.decoded_eegs;
        
        % Load the z-scored EEGs
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub', num2str(subNo), '.mat'));
        zscore_eegs = zscore_eegs_file.zscore_data;
        
        for chno = 1:channelNum
            % Compute correlation coefficient between RBM decoded EEGs and z-scored EEGs for each channel
            R = corrcoef(decoded_eegs_rbm(:, chno), zscore_eegs(:, chno));
            corr_chs(j, subNo, chno) = R(1, 2);
        end
        
        % Display the correlation coefficients for the current subject and latent dimension
        corr_chs(j, subNo, :)
    end
end

% Save the calculated correlation coefficients to a .mat file
fileName = 'C:\Users\LEGION\Desktop\LatentFactorDecodingEEG-master\calclulated_reconstruction_performance\rbm_reconstruction_corr_Rvalue.mat';
save(fileName, 'corr_chs', '-v7.3');
