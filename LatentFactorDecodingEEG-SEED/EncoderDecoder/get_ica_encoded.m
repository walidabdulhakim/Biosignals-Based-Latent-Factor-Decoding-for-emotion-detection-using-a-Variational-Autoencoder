
subNum=15;
channelNum=62;
zscore=1;

latdim = 16;

for subNo=15
    for latent_dim = 1:latdim
        zscore_eegs_file = load(strcat('D:\Processed SEED\normalize_zscore\s',num2str(subNo),'.mat')); 
        zscore_eegs = zscore_eegs_file.data(:,1:channelNum)';
        disp(strcat('subNo: ',num2str(subNo),' latentdim: ', num2str(latent_dim)));
         %fast ICA A:Mixing matrix W: Seperating matrix
        [ICs, A, W] = fastica(zscore_eegs,'numOfIC',latent_dim,'verbose','off');
        decoded_eegs = A*ICs;
        encoded_eegs = ICs;
        %save data
        fileName = strcat('D:\Processed SEED\ICA\encoded_eegs_ica\encoded_eegs_ica_sub',num2str(subNo),'_latentdim',num2str(latent_dim));
        save(fileName,'encoded_eegs','-v7.3');

    end
end