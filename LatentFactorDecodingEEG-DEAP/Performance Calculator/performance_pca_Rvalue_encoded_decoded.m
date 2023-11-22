subNum = 32;
channelNum = 32;
testNum = 10;
latdimNum = 16;

corr_chs = zeros(latdimNum, subNum, channelNum);

for subNo = 1:subNum
    zscore_eegs_file = load(strcat('D:\Processed DEAP DATA\normalize_zscore\sub', num2str(subNo), '.mat'));
    zscore_eegs = zscore_eegs_file.zscore_data(:, 1:32)';
    
    for latdim = 1:latdimNum
        disp(strcat('subNo: ', num2str(subNo), ' latdim: ', num2str(latdim)));
        
        % Perform PCA and reconstruct the EEG data
        [residuals, reconstructed] = pcares(zscore_eegs, latdim);
        decoded_eegs_pca = reconstructed;
        
        for chno1 = 1:channelNum
            max_r = 0;
            
            for chno2 = 1:channelNum
                % Compute correlation coefficient between PCA decoded EEGs and z-scored EEGs for each channel pair
                R = corrcoef(decoded_eegs_pca(chno1, :), zscore_eegs(chno2, :));
                r = R(1, 2);
                
                if r > max_r
                    max_r = r;
                end
            end
            
            corr_chs(latdim, subNo, chno1) = max_r;
        end
        
        % Display the correlation coefficients for the current subject and latent dimension
        corr_chs(latdim, subNo, :)
    end
end

% Save the calculated correlation coefficients to a .mat file
fileName = 'C:\Users\LEGION\Desktop\LatentFactorDecodingEEG-master\calclulated_reconstruction_performance\ae_reconstruction_corr_Rvalue.mat';
save(fileName, 'corr_chs', '-v7.3');
