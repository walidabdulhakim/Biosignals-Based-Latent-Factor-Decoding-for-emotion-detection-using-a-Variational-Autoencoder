
subNum=15;
channelNum=62;
zscore=1;

latdim = 16;

for subNo=1:subNum
    for latent_dim = 1:latdim 
        zscore_eegs_file = load(strcat('D:\Processed SEED\normalize_zscore\s',num2str(subNo),'.mat')); 
        zscore_eegs = zscore_eegs_file.data(:,1:channelNum)';
        disp(strcat('subNo: ',num2str(subNo),' latentdim: ', num2str(latent_dim)));
        
        X = zscore_eegs.';
        [coeff, score, latent] = pca(X);
        encoded_eegs = score(:,1:latent_dim);
%         [pcaData, coeff] = fastPCA(X,latdim);
%         encoded_eegs = pcaData;
        %save data
        fileName = strcat('D:\Processed SEED\PCA\encoded_eegs_pca\encoded_eegs_pca_sub',num2str(subNo),'_latedtdim',num2str(latent_dim));
        save(fileName,'encoded_eegs','-v7.3');

    end
end
