latent_dims = 1:1:16;
subNum = 32;
channelNum = 32;
zscore = 1;
corr_chs = zeros(length(latent_dims), subNum, channelNum);

for subNo = 1:subNum
    for j = 1:length(latent_dims)
        disp(strcat('subNo: ', num2str(subNo), ' latentdim: ', num2str(latent_dims(j))));
        
        % Load z-scored EEGs
        zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub', num2str(subNo), '.mat'));
        zscore_eegs = zscore_eegs_file.zscore_data(:, 1:32)';
        
        % Perform Independent Component Analysis (ICA)
        [ICs, A, W] = fastica(zscore_eegs, 'numOfIC', latent_dims(j), 'verbose', 'off');
        decoded_eegs = A * ICs;
        
        for chno1 = 1:channelNum
            max_r = 0;
            
            for chno2 = 1:channelNum
                % Compute correlation coefficient between ICA decoded EEGs and z-scored EEGs for each channel pair
                R = corrcoef(decoded_eegs(chno1, :), zscore_eegs(chno2, :));
                r = R(1, 2);
                
                if r > max_r
                    max_r = r;
                end
            end
            
            corr_chs(j, subNo, chno1) = max_r;
        end
    end
end

% Save the calculated correlation coefficients to a .mat file
fileName = 'C:\Users\LEGION\Desktop\graduation codes\LatentFactorDecodingEEG-master\calclulated_reconstruction_performance\ica_reconstruction_corr_Rvalue.mat';
save(fileName, 'corr_chs', '-v7.3');
