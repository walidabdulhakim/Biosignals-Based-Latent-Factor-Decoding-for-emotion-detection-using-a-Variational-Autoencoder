

%load('lstm_personal_performance_vae.mat');
%load('lstm_personal_performance_ae.mat');
%load('lstm_personal_performance_ica.mat');
%load('lstm_personal_performance_pca.mat');
%load('lstm_personal_performance_rbm.mat');

%load('gru_personal_performance_vae.mat');
%load('gru_personal_performance_ae.mat');
%load('gru_personal_performance_ica.mat');
%load('gru_personal_performance_pca.mat');
load('gru_personal_performance_rbm.mat');

% Calculate the mean of each variable
mean_var1 = mean(test_f1);
mean_var2 = mean(test_acc);
mean_var3 = mean(train_f1);
mean_var4 = mean(train_acc);

% Create a time vector (assuming data is sampled at 128 Hz)
Sub = (1:15)*200;

% Plot the variables
figure;
subplot(4,1,1);
stem(Sub, test_f1);
hold on;
plot([0 16], [mean_var1 mean_var1], 'r--');
hold off;
xlabel('Subject');
ylabel('test f1');
text(1,mean_var1+0.01,sprintf('Mean: %.3f',mean_var1));

subplot(4,1,2);
stem(Sub, test_acc);
hold on;
plot([0 16], [mean_var2 mean_var2], 'r--');
hold off;
xlabel('Subject');
ylabel('test acc');
text(1,mean_var2+0.01,sprintf('Mean: %.3f',mean_var2));

subplot(4,1,3);
stem(Sub, train_f1);
hold on;
plot([0 16], [mean_var3 mean_var3], 'r--');
hold off;
xlabel('Subject');
ylabel('train f1');
text(1,mean_var3+0.01,sprintf('Mean: %.3f',mean_var3));

subplot(4,1,4);
stem(Sub, train_acc);
hold on;
plot([0 16], [mean_var4 mean_var4], 'r--');
hold off;
xlabel('Subject');
ylabel('train acc');
text(1,mean_var4+0.01,sprintf('Mean: %.3f',mean_var4));


