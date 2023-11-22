
subNum=15;
channelNum=62;
latent_dims = 1:1:16;




corr_chs = zeros(length(latent_dims),subNum,channelNum);
for subNo=1:subNum
    for j=1:length(latent_dims)
    filename1 = strcat(strcat('D:\Processed SEED DATA\decoded_eegs_1rbm\decoded_eegs_1rbm_sub',num2str(subNo),'_latentdim',num2str(j),'.mat'));
    decoded_eegs_rbm_file = load(filename1);
    decoded_eegs_rbm = decoded_eegs_rbm_file.decoded_eegs;
    zscore_eegs_file = load(strcat('D:\Processed SEED DATA\normalize_zscore\s1',num2str(subNo),'.mat'));
    zscore_eegs = zscore_eegs_file.data;
    for chno=1:channelNum
        R=corrcoef(decoded_eegs_rbm(:,chno), zscore_eegs(:,chno));
        corr_chs(j,subNo,chno)=R(1,2);
    end
    corr_chs(j,subNo,:)
    end
end
%mean(corr_chs)

fileName = strcat('D:\Processed SEED DATA\calclulated_reconstruction_performance\rbm_reconstruction_corr_Rvalue.mat');
        save(fileName,'corr_chs','-v7.3');