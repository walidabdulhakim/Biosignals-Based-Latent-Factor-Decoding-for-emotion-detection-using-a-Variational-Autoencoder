subNum=15;
channelNum=62;
latdimNum=16;


corr_chs = zeros(latdimNum,subNum,channelNum);
for latdim=1:latdimNum
    for subNo=1:subNum
        filename1 = strcat(strcat('D:\VAE Experiment\SEED\decoded_eegs_vae\decoded_eegs_vae_sub',num2str(subNo),'_latentdim',num2str(latdim),'.mat'));
        decoded_eegs_ae_file = load(filename1);
        decoded_eegs_ae = decoded_eegs_ae_file.decoded_eegs;
        zscore_eegs_file = load(strcat('D:\Processed SEED DATA\normalize_zscore\s1',num2str(subNo),'.mat'));
        zscore_eegs = zscore_eegs_file.data;
        for chno=1:channelNum
            disp(strcat('latdim: ', num2str(latdim), ' subNo: ', num2str(subNo), ' chNo: ', num2str(chno)));
            R=corrcoef(decoded_eegs_ae(:,chno), zscore_eegs(:,chno));
            corr_chs(latdim,subNo,chno)=R(1,2);
        end
    end 
end

ms = zeros(1,latdimNum);
for latdim=1:latdimNum
   latdim_performance = squeeze(corr_chs(latdim,:,:));
   m = mean(mean(latdim_performance));
   ms(1,latdim)=m;
end

fileName = strcat('D:\Processed SEED DATA\calclulated_reconstruction_performance\vae_reconstruction_corr_Rvalue.mat');
        save(fileName,'corr_chs','-v7.3');