

subNum = 15;
expNum = 1; %每个被试实验3次
trialNum = 30;
channelNum = 62;
fs = 200;
trialTime = 20;
trialL = fs*trialTime;
signalL = trialL*channelNum;

% 分别对每个channel在40段视频里的数据进行normalization，scale到0~1
% 这种normalize的方式能保留各个channel在不同刺激下的幅值差异，同时还能消除掉不同channel的幅值差异，同时还能降低被试与被试间的差异
for expNo=1:expNum
    for subNo=1:subNum
        data = zeros(channelNum, signalL);
      
        filePath = strcat('C:\\Users\\LEGION\\Desktop\\1\\',num2str(subNo));
        
        datFile = load(filePath);
        trialNames = fieldnames(datFile); % 取出结构体内所有字段
        channel_data = zeros(channelNum, trialL);
        
            for trialNo = 1:trialNum
                trialName = trialNames{trialNo};
                trialData = getfield(datFile, trialName);
                disp(trialName);
                for channelNo = 1:channelNum
                 disp(strcat('start processing sub ', num2str(subNo), ' experiment ', num2str(expNo), ' channel ', num2str(channelNo), ' trial ', num2str(trialNo)));
                 channelSignal = trialData(channelNo, : );
                
                 % 取最中间的20s部分作为目标信号，因为考虑到采样率200，需要降低抽取特征计算代价，即缩短信号长度
                 length = size(channelSignal, 2);
                 l_center = round(length/2);
                 centerSignal = channelSignal(l_center-fs*10+1:l_center+fs*10);
                
                 channel_data(channelNo, :) = centerSignal;
                 startIndex = (channelNo-1)*trialL+1;
                 endIndex = channelNo*trialL;
                 data(:, startIndex:endIndex) = channel_data;
                       
                end
                
            
            end
        
        data = zscore(data');
        %data = data.';
        
        % 将该被试的data保存起来
        fileName = strcat('C:\Users\LEGION\Desktop\normalized\s',num2str(subNo));
        save(fileName, 'data', '-v7.3');
        disp(strcat('end!subject ', num2str(subNo)));
    end
    
end