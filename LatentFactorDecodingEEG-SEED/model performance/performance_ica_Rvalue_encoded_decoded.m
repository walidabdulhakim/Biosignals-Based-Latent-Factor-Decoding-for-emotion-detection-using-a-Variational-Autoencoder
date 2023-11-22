
latent_dims = 16;
%inter_dims = [16, 64, 128, 256];
subNum=15;
channelNum=62;
zscore=1;

corr_chs = zeros(latent_dims,subNum,channelNum);

for subNo=1:subNum
    for j=1:latent_dims
        disp(strcat('subNo: ',num2str(subNo),' latentdim: ', num2str(j)));
        zscore_eegs_file = load(strcat('D:\Processed SEED\normalize_zscore\s',num2str(subNo),'.mat'));
        zscore_eegs = zscore_eegs_file.data(:,1:channelNum)';
         %fast ICA A:Mixing matrix W: Seperating matrix
        [ICs, A, W] = fastica(zscore_eegs,'numOfIC',j,'verbose','off');
        decoded_eegs = A*ICs;
        %decoded_eegs_ica = A*(W*zscore_eegs);        
        for chno1=1:channelNum
            max_r = 0;
            for chno2=1:channelNum
                 R=corrcoef(decoded_eegs(chno1,:), zscore_eegs(chno2,:));
                 r=R(1,2);
                 if r>max_r
                     max_r=r;
                 end    
            end
            corr_chs(j,subNo,chno1)=max_r;
        end
           
     end
 end

%各个被试表现最佳的结果及其对应的网络结构
%best_sub_corrs = zeros(1,subNum);
%best_latent_dims = zeros(1,subNum);
%for subNo=1:subNum
%   sub_corr_chs = squeeze(corr_chs(subNo,:,:));
%    mean_sub_corr_chs = mean(sub_corr_chs,2);
%    [best_sub_corr,best_latent_dim] = max(mean_sub_corr_chs);
%    best_sub_corrs(1,subNo)=best_sub_corr;
%    best_latent_dims(1,subNo)=best_latent_dim;
%end
%best_sub_corrs
%mean_best_sub_corr = mean(best_sub_corrs);
%best_latent_dims
%mean_best_sub_corr

%平均被试表现最佳的结果及其对应的网络结构
%latentdim_mean_corrs = zeros(1,length(latent_dims));
%for latent_dim=1:16
%    lat_corr_chs = squeeze(corr_chs(:,latent_dim,:));
%    latentdim_mean_corrs(latent_dim) = mean(mean(lat_corr_chs));
%end
%[best_latentdim_mean_corr,best_latent_dim] = max(latentdim_mean_corrs);
%best_latent_dim
%best_latentdim_mean_corr


%mean(corr_chs)

fileName = strcat('D:\Processed SEED\Calculated\ica_reconstruction_corr_Rvalue.mat');
        save(fileName,'corr_chs','-v7.3');

