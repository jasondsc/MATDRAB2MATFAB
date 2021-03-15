
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                Class 3
%                            Signal Processing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

%% Simulating data in MATLAB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% you can use the rand, randn, and randi functions
randn(1) % returns random number taken from normal dist
randi(1) % retruns random number of maximum i
rand(1) % returns random number between 0 and 1

randn(10)
randn(1,10)
randi(100, [1,10])
randi(100, [10])
rand(10)
rand(1,10)

% simulate oscilatory data with sin
fs = 1000; % Sampling frequency (samples per second) 
dt = 1/fs; % seconds per sample 
StopTime = 3; % seconds 
t = (0:dt:StopTime)'; % seconds 
F = 60; % Sine wave frequency (hertz) 
data = sin(2*pi*F*t);
figure
plot(t,data)

% add noise
data = sin(2*pi*F*t)+rand(length(data),1);
plot(t,data)

% nest signals 
F2=10;
data = sin(2*pi*F*t)+rand(length(data),1) +  sin(2*pi*F2*t);
plot(t,data)


% exercise 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%try generating random numbers between -1 and 1 using rand

% try generating random doubles between 1 and 10 using rand

% try generating random doubles between 5 and 25 using randi

% generate a oscilatory signal that monotonically decreases over time



%% Data Smoothing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./justafolderwithdata/raw_spike_data.mat') % load spiking data recorded from electrode array
sr=44100; % sampling rate of the electrode array
% plot data
figure
subplot(2,1,1)
plot(raw_spike_data) % plot data to visualize spikes and time series
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording '); 

% Lets try smoothig out some of those high frequency components 
subplot(2,1,2)
plot(smooth(raw_spike_data)) 
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


% detrend works by removing linear trend in data 
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

%% Normalizing data and removing outliers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate random data
temp_data=randi(100,10);
% try removing the data's mean
temp_data-mean(temp_data)
temp_data-mean(temp_data,2)
temp_data-mean(mean(temp_data))

% normalize 
normalize(temp_data)
normalize(temp_data, 'range')
normalize(temp_data, 'center')
% normalize does the equevelant of ....
(temp_data-mean(temp_data))./std(temp_data)

% filloutliers can remove data that does not seem to belong
data = sin(2*pi*F*t)+rand(length(data),1) +  sin(2*pi*F2*t)
data(randi(length(data),1,50))=100;
plot(t,data)           
plot(filloutliers(data, 'linear'))


%% Spectral Power, FFTs, and PSDs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%% Filtering Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sr=44100; % sampling rate of the electrode array
figure
subplot(2,1,1)
plot(raw_spike_data) % plot data to visualize spikes and time series
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording '); % clear slow drifts in data that need to be removed

% create and apply bandpass butter worth filter
[b,a] = butter(3,[500/(sr/2) 8000/(sr/2)], 'bandpass'); % create butterworth filter 
filtered_data=filter(b,a,raw_spike_data); % apply butter worth filter 

xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')


% create and apply lowpass butter worth filter
[b,a] = butter(3,[100/(sr/2)], 'low'); % create butterworth filter 
filtered_data=filter(b,a,raw_spike_data); % apply butter worth filter 

% plot filtered data with unfiltered data 
subplot(2,1,2)
plot(filtered_data)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');


% create and apply highpass butter worth filter
[b,a] = butter(3,[500/(sr/2)], 'high'); % create butterworth filter 
filtered_data=filter(b,a,raw_spike_data); % apply butter worth filter 

% plot filtered data with unfiltered data 
subplot(2,1,2)
plot(filtered_data)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');


% lets try FIR filters 
load chirp % load data
Fs=8192;
t = (0:length(y)-1)/Fs;

%create filter for data
bhi = fir1(34,100/Fs,'low');
freqz(bhi,1)
outhi = filter(bhi,1,y); % filter data

% plot results 
figure
subplot(2,1,1)
plot(t,y)
title('Original Signal')
ys = ylim;

subplot(2,1,2)
plot(t,outhi)
title('Highpass Filtered Signal')
xlabel('Time (s)')
ylim(ys)

soundsc(y)
pause(2)
soundsc(outhi)


bhi = fir1(34,[100/Fs 1000/Fs ],'bandpass');
freqz(bhi,1)

outhi = filter(bhi,1,y);
figure
subplot(2,1,1)
plot(t,y)
title('Original Signal')
ys = ylim;

subplot(2,1,2)
plot(t,outhi)
title('Highpass Filtered Signal')
xlabel('Time (s)')
ylim(ys)

soundsc(y)
pause(2)
soundsc(outhi)


bhi = fir1(34,[100/Fs 5000/Fs ],'bandpass');
freqz(bhi,1)

outhi = filter(bhi,1,y);
figure
subplot(2,1,1)
plot(t,y)
title('Original Signal')
ys = ylim;

subplot(2,1,2)
plot(t,outhi)
title('Highpass Filtered Signal')
xlabel('Time (s)')
ylim(ys)

soundsc(y)
pause(2)
soundsc(outhi)


% exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%try highpass and lowpass filtering the electrophysiology data into
% fast and slow oscilatory components. Try adding them back together and
% see what happens, try moving around the filter frequency. What could be
% causing this?


%% Hilbert Transformation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate oscillatory data
Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1000;             % Length of signal
t = (0:L-1)*T;        % Time vector

data=zeros(1000,1);
data = sin(2*pi*F*t) + rand(length(data),1)' +  sin(2*pi*F2*t) + 1/length(data)*[1:length(data)];
figure
plot(t,data)

% get the hilbert transform of data
hil=hilbert(data); % notice that the data is stored as complex not int or double

subplot(1,2,1)
plot(t,data)
hold on
plot(t,abs(hil), '-r')
subplot(1,2,2)
plot(t,angle(hil))


data = sin(2*pi*2*t).*sin(2*pi*F*t) ;
figure
plot(t,data)

hil=hilbert(data);

subplot(1,2,1)
plot(t,data)
hold on
plot(t,abs(hil), '-r')
subplot(1,2,2)
plot(t,angle(hil))

hil=hilbert(raw_spike_data);
figure
subplot(1,2,1)
plot(raw_spike_data)
hold on
plot(abs(hil), '-r')
subplot(1,2,2)
plot(angle(hil))


%% Finding peaks 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% threshold data to find peaks (i.e., spikes) with MATLAB's findpeaks
[pks,locs]=findpeaks(-1*filtered_data, 'MinPeakHeight',0.3); 
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

% MinPeakProminence
[pks,locs]=findpeaks(-1*filtered_data, 'MinPeakProminence',0.3); 

% Plot peaks found
figure 
plot(filtered_data)
hold on;
scatter(locs,-1*pks)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');

% Threshold
[pks,locs]=findpeaks(-1*filtered_data, 'Threshold',0.1); 

% Plot peaks found
figure 
plot(filtered_data)
hold on;
scatter(locs,-1*pks)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');

% exercise 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%try using the mean, median, and std of the data determine the cutoff for
%detecting peaks.