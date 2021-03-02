
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% class 3

%% simulating data

randn(1)
randi(1)
rand(1)

randn(10)
randn(1,10)
randi(100, [1,10])
randi(100, [10])
rand(10)
rand(1,10)

% simulate oscilatory data

fs = 1000; % Sampling frequency (samples per second) 
 dt = 1/fs; % seconds per sample 
 StopTime = 3; % seconds 
 t = (0:dt:StopTime)'; % seconds 
 F = 60; % Sine wave frequency (hertz) 
 data = sin(2*pi*F*t);
 plot(t,data)

% add noise
 data = sin(2*pi*F*t)+rand(length(data),1);
 plot(t,data)
 
 % nest signals 
 F2=10;
data = sin(2*pi*F*t)+rand(length(data),1) +  sin(2*pi*F2*t);
 plot(t,data)


%% Smoothing 

clear all
close all
load('./justafolderwithdata/raw_spike_data.mat') % load data spiking data recorded from electrode array
sr=44100; % sampling rate of the electrode array
figure
subplot(2,1,1)
plot(raw_spike_data) % plot data to visualize spikes and time series
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording '); % clear slow drifts in data that need 
                                                                                     %to be removed with a filter before clustering
subplot(2,1,2)
plot(smooth(raw_spike_data)) % plot data to visualize spikes and time series
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording ');

subplot(2,1,2)
plot(smooth(raw_spike_data, 20)) % plot data to visualize spikes and time series
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording ');
                                                                                     
subplot(2,1,2)
plot(smoothdata(raw_spike_data)) % plot data to visualize spikes and time series
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording ');
                                                                                     

fs = 1000; % Sampling frequency (samples per second) 
dt = 1/fs; % seconds per sample 
StopTime = 3; % seconds 
t = (0:dt:StopTime)'; % seconds 
F = 60; % Sine wave frequency (hertz) 
F2=10;
data = sin(2*pi*F*t);
data = sin(2*pi*F*t)+rand(length(data),1) +  sin(2*pi*F2*t) + 1/length(data)*[1:length(data)]';
plot(t,data)
plot(t,detrend(data))

%% Normalize data 

temp_data=randi(100,10);

temp_data-mean(temp_data)
temp_data-mean(temp_data,2)
temp_data-mean(mean(temp_data))

normalize(temp_data)

(temp_data-mean(temp_data))./std(temp_data)

data = sin(2*pi*F*t)+rand(length(data),1) +  sin(2*pi*F2*t)
data(randi(length(data),1,50))=100;
plot(t,data)           
plot(filloutliers(data, 'linear'))


%% FFT

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 10000;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = 0.7*sin(2*pi*33*t) + sin(2*pi*153*t) + 3*sin(2*pi*222*t);
X = S + 2*randn(size(t));

Y = fft(X);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);


f = Fs*(0:(L/2))/L;
figure
subplot(1,2,1)
plot(X)
subplot(1,2,2)
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')


S = 0.7*sin(2*pi*33*t) + sin(2*pi*153*t) + 10*sin(2*pi*222*t);
X = S + 2*randn(size(t));

Y = fft(X);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
figure
subplot(1,2,1)
plot(X)
subplot(1,2,2)
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

pwelch(X)

S = 0.7*sin(2*pi*33*t) + sin(2*pi*153*t) + 10*sin(2*pi*222*t);
X = S + 2*randn(size(t));

[pxx,f] = pwelch(X,500,300,500,Fs);
figure
plot(f,10*log10(pxx))

xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')

[pxx,f] = pwelch(X,1000,500,500,Fs);
figure
plot(f,10*log10(pxx))

xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')


[pxx,f] = pwelch(X,100,50,500,Fs);
figure
plot(f,10*log10(pxx))

xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')


%% filter data 
% create and apply filter butter worth filter
[b,a] = butter(3,[500/(sr/2) 8000/(sr/2)], 'bandpass'); % create butterworth filter 
filtered_data=filter(b,a,raw_spike_data); % apply butter worth filter 

% plot filtered data with unfiltered data 
subplot(2,1,2)
plot(filtered_data)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');

% threshold data to find peaks (i.e., spikes) with MATLAB's findpeaks
[pks,locs]=findpeaks(-1*filtered_data, 'MinPeakHeight',3.5*std(filtered_data)); 
% TIP input the inverse of the data (i.e. -1*) as findpeaks looks
% for local maxima and not minima

% Plot peaks found
figure 
plot(filtered_data)
hold on;
scatter(locs,-1*pks)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');

% extract deisred window [-20 to 43] samples around a peak
for i =1:length(locs)
peaks(i,:)=filtered_data(locs(i)-20:locs(i)+43);
end

figure
plot(-20:43, peaks')
axis tight
xlabel('time (number of samples)');
ylabel('Micro Volts ');
title('Extracted Timeseries of Peaks ');



%% hilbert 
data=zeros(3001,1);
data = sin(2*pi*F*t) + rand(length(data),1) +  sin(2*pi*F2*t) + 1/length(data)*[1:length(data)]';
 plot(t,data)
 
 hil=hilbert(data);
 
 figure
 subplot(1,2,1)
 plot(t,data)
 hold on
  plot(t,abs(hil), '-r')
  subplot(1,2,2)
 plot(t,angle(hil))


data = sin(2*pi*2*t).*sin(2*pi*F*t) ;
 plot(t,data)
 
 hil=hilbert(data);
 
 figure
 subplot(1,2,1)
 plot(t,data)
 hold on
  plot(t,abs(hil), '-r')
  subplot(1,2,2)
 plot(t,angle(hil))
 

%%%%%%%%%%%%%%%%%%%%%%%%%%
%% class 5
clear all
close all

%% PCA example with image
image=imread('./WhoAmI.jpg');
%image=imread('./ladygaga4.jpg');


image=mean(double(image),3);
figure
subplot(3,3,9)
imagesc(image)
colormap(gray)
axis off;


[coeff, score, latent, tsquared, explained, mu]=pca(image);

subplot(3,3,1)
imagesc(score(:,1)*coeff(:,1)')
colormap(gray)
axis off;
subplot(3,3,2)
imagesc(score(:,1:2)*coeff(:,1:2)')
colormap(gray)
axis off;
subplot(3,3,3)
imagesc(score(:,1:5)*coeff(:,1:5)')
colormap(gray)
axis off;
subplot(3,3,4)
imagesc(score(:,1:10)*coeff(:,1:10)')
colormap(gray)
axis off;
subplot(3,3,5)
imagesc(score(:,1:20)*coeff(:,1:20)')
colormap(gray)
axis off;
subplot(3,3,6)
imagesc(score(:,1:30)*coeff(:,1:30)')
colormap(gray)
axis off;
subplot(3,3,7)
imagesc(score(:,1:50)*coeff(:,1:50)')
colormap(gray)
axis off;
subplot(3,3,8)
imagesc(score(:,1:100)*coeff(:,1:100)')
colormap(gray)
axis off;


%% ICA

figure
subplot(3,4,12)
imagesc(image)
colormap(gray)
axis off;

ica_out=rica(image,50);

t=transform(ica_out,image);


subplot(3,4,1)
imagesc(t(:,1)*ica_out.TransformWeights(:,1)')
colormap(gray)
axis off

subplot(3,4,2)
imagesc(t(:,1:2)*ica_out.TransformWeights(:,1:2)')
colormap(gray)
axis off;

subplot(3,4,3)
imagesc(t(:,1:3)*ica_out.TransformWeights(:,1:3)')
colormap(gray)
axis off;


subplot(3,4,4)
imagesc(t(:,1:5)*ica_out.TransformWeights(:,1:5)')
colormap(gray)
axis off;

subplot(3,4,5)
imagesc(t(:,1:10)*ica_out.TransformWeights(:,1:10)')
colormap(gray)
axis off;

subplot(3,4,6)
imagesc(t(:,1:20)*ica_out.TransformWeights(:,1:20)')
colormap(gray)
axis off;

subplot(3,4,7)
imagesc(t(:,1:25)*ica_out.TransformWeights(:,1:25)')
colormap(gray)
axis off;

subplot(3,4,8)
imagesc(t(:,1:30)*ica_out.TransformWeights(:,1:30)')
colormap(gray)
axis off;

subplot(3,4,9)
imagesc(t(:,1:35)*ica_out.TransformWeights(:,1:35)')
colormap(gray)
axis off;

subplot(3,4,10)
imagesc(t(:,1:40)*ica_out.TransformWeights(:,1:40)')
colormap(gray)
axis off;

subplot(3,4,11)
imagesc(t(:,1:45)*ica_out.TransformWeights(:,1:45)')
colormap(gray)
axis off;



%%
rng default

files = {'chirp.mat'
        'gong.mat'
        'handel.mat'
        'laughter.mat'
        'splat.mat'
        'train.mat'};

S = zeros(10000,6);
for i = 1:6
    test     = load(files{i});
    y        = test.y(1:10000,1);
    S(:,i)   = y;
end

mixdata = S*randn(6) + randn(1,6);

figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(mixdata(:,i))
    title(['Mix ',num2str(i)])
end

% play sound
soundsc(S(:,1));
pause(2)
soundsc(mixdata(:,1))

% run ica
mixdata = prewhiten(mixdata);
q = 6;
Mdl = rica(mixdata,q,'NonGaussianityIndicator',ones(6,1));

unmixed = transform(Mdl,mixdata);

figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(unmixed(:,i))
    title(['Unmix ',num2str(i)])
end

% play sound
soundsc(S(:,1));
pause(2)
soundsc(unmixed(:,1))




%%                                           K- means
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all
close all
load('./justafolderwithdata/raw_spike_data.mat') % load data spiking data recorded from electrode array
sr=44100; % sampling rate of the electrode array
figure
subplot(2,1,1)
plot(raw_spike_data) % plot data to visualize spikes and time series
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording '); % clear slow drifts in data that need 
                                                                                     %to be removed with a filter before clustering



% create and apply filter butter worth filter
[b,a] = butter(3,[500/(sr/2) 8000/(sr/2)], 'bandpass'); % create butterworth filter 
filtered_data=filter(b,a,raw_spike_data); % apply butter worth filter 

% plot filtered data with unfiltered data 
subplot(2,1,2)
plot(filtered_data)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');

% threshold data to find peaks (i.e., spikes) with MATLAB's findpeaks
[pks,locs]=findpeaks(-1*filtered_data, 'MinPeakHeight',3.5*std(filtered_data)); 
% TIP input the inverse of the data (i.e. -1*) as findpeaks looks
% for local maxima and not minima

% Plot peaks found
figure 
plot(filtered_data)
hold on;
scatter(locs,-1*pks)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');

% extract deisred window [-20 to 43] samples around a peak
for i =1:length(locs)
peaks(i,:)=filtered_data(locs(i)-20:locs(i)+43);
end

figure
plot(-20:43, peaks')
axis tight
xlabel('time (number of samples)');
ylabel('Micro Volts ');
title('Extracted Timeseries of Peaks ');

% run PCA over peaks to extract features; how many components are
% visbale? 
[coeff,score,latent,~,explained] = pca(peaks);
figure
plot(explained, '-*'); % plot percent var explained
xlabel("Component Number")
ylabel("Percent Variance Explained (%)")

% lets plot first two components and see how the data scatters
figure
scatter(score(:,1), score(:,2), '.')

% now that we have extracted features from the data let us see how they
% cluster together

% Run K-means clustering 
opts = statset('Display','final');
numclust=10; % max number of k
ctrs = zeros(size(peaks,1), numclust);
ids = zeros(size(peaks,1), numclust);
sumd= zeros(1,numclust);

for j=1:numclust
[ids(:,j)] = kmeans(peaks, j, 'Replicates', 20, 'Options', opts);

end

% find the optimal solution for the best number of clusters

eva_cali = evalclusters(peaks,ids,'CalinskiHarabasz') % evaluate the best number of k clusters
eva_sil = evalclusters(peaks,ids,'silhouette') % evaluate the best number of k clusters
eva_Dav = evalclusters(peaks,ids,'DaviesBouldin') % evaluate the best number of clusters

% all three methods returned different numbers, plot the silhouettes
figure
silhouette(peaks,ids(:,2))
figure
silhouette(peaks,ids(:,3))
figure
silhouette(peaks,ids(:,4))

% plot 3 cluster solution in PCA space
[ids_opt, crit_opt] = kmeans(score, 3, 'Replicates', 20, 'Options', opts); % 3 cluster optimal solution

figure
scatter(score(ids_opt==1,1), score(ids_opt==1,2), '.g')
hold on;
scatter(score(ids_opt==2,1), score(ids_opt==2,2), '.m')
hold on;
scatter(score(ids_opt==3,1), score(ids_opt==3,2), '.c')
hold on;
scatter(crit_opt(1:3,1), crit_opt(1:3,2), 200, 'kX')
hold on;
scatter(crit_opt(1:3,1), crit_opt(1:3,2), 200, 'kd')


%%%%%%% REMINDER TO CHNAGE COLOUR OF PLOTS CANNOT SEE



%Drag race clustering example: 





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% class 6 

   X = randn(8,8);
   Y = randn(8,8);
   
   corr(X,Y)
   corr2(X,Y)
   corrcoef(X,Y)
   
   
