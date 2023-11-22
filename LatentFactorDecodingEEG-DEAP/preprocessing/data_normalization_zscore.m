subNum = 32;
trialNum = 40;
channelNum = 40;
fs = 128;
trialTime = 63; 
trialL = fs * trialTime;

for subNo = 1:subNum
    
    % Generate the file path for loading the subject data
    if subNo < 10
        filePath = strcat('D:\DEAP DATA\s0', num2str(subNo), '.mat');
    else
        filePath = strcat('D:\DEAP DATA\s', num2str(subNo), '.mat');
    end
    
    % Load the subject data from the file
    datFile = load(filePath);
    subData = datFile.data;
    
    % Reshape the subject data for z-scoring
    reshape_subData = zeros(channelNum, trialNum * trialL);
    for channelNo = 1:channelNum
        for trialNo = 1:trialNum
            ch_tr_signal = subData(trialNo, channelNo, :);
            reshape_subData(channelNo, (trialNo - 1) * trialL + 1:trialNo * trialL) = ch_tr_signal;
        end
    end
    
    % Z-score the reshaped subject data
    zscore_data = zscore(reshape_subData');
    
    % Generate the file name for saving the z-scored data
    fileName = strcat('D:\Processed DEAP DATA\normalize_zscore\sub', num2str(subNo));
    
    % Save the z-scored data
    save(fileName, 'zscore_data', '-v7.3');
    
    % Display a message indicating the completion of z-scoring for the current subject
    disp(strcat('Z-scoring complete for subject ', num2str(subNo)));
end
