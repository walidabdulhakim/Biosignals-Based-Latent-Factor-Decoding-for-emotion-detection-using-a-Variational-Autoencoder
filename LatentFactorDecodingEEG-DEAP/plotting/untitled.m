% Load the .mat file
%load('lstm_personal_performance_1ae_arousal_subcross.mat');
%load('lstm_personal_performance_vae_arousal_subcross.mat');
%load('lstm_personal_performance_pca_arousal_subcross.mat');
%load('lstm_personal_performance_ica_arousal_subcross.mat');
%load('lstm_personal_performance_rbm_arousal_subcross.mat');

%load('lstm_personal_performance_1ae_valence_subcross.mat');
%load('lstm_personal_performance_vae_valence_subcross.mat');
%load('lstm_personal_performance_pca_valence_subcross.mat');
%load('lstm_personal_performance_ica_valence_subcross.mat');
%load('lstm_personal_performance_rbm_valence_subcross.mat');

%load('lstm_personal_performance_vae_weighted_valence_subcross.mat');
%load('lstm_personal_performance_vae_weighted_arousal_subcross.mat');

%load('gru_personal_performance_vae_weighted_valence_subcross.mat');
%load('gru_personal_performance_vae_weighted_arousal_subcross.mat');

%load('gru_personal_performance_rbm_weighted_valence_subcross.mat');
%load('gru_personal_performance_vae_weighted_arousal_subcross.mat');


% Calculate the mean of each variable
mean_var1 = mean(test_f1);
mean_var2 = mean(test_acc);
mean_var3 = mean(train_f1);
mean_var4 = mean(train_acc);

% Create a time vector (assuming data is sampled at 128 Hz)
Sub = (1:32)*128;

% Plot the variables
figure;
subplot(4,1,1);
stem(Sub, test_f1);
hold on;
plot([0 33], [mean_var1 mean_var1], 'r--');
hold off;
xlabel('Subject');
ylabel('test f1');
text(1,mean_var1+0.01,sprintf('Mean: %.3f',mean_var1));

subplot(4,1,2);
stem(Sub, test_acc);
hold on;
plot([0 33], [mean_var2 mean_var2], 'r--');
hold off;
xlabel('Subject');
ylabel('test acc');
text(1,mean_var2+0.01,sprintf('Mean: %.3f',mean_var2));

subplot(4,1,3);
stem(Sub, train_f1);
hold on;
plot([0 33], [mean_var3 mean_var3], 'r--');
hold off;
xlabel('Subject');
ylabel('train f1');
text(1,mean_var3+0.01,sprintf('Mean: %.3f',mean_var3));

subplot(4,1,4);
stem(Sub, train_acc);
hold on;
plot([0 33], [mean_var4 mean_var4], 'r--');
hold off;
xlabel('Subject');
ylabel('train acc');
text(1,mean_var4+0.01,sprintf('Mean: %.3f',mean_var4));


