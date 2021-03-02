clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                        INTRO TO DATA TYPES/ STRUCTURE
%           
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make an array
ar=[]; % empty array
ar=[1,2,3,4,5,6,7]; % array of ints
ar=[1,2,3;4,5,6;7,8,9]; % matrix array of ints
z=zeros(10,10);
z=z+10;
z=z-9;
z=z*3;

z.*z % IS DIFFERENT THAN ' * '

z*z % THIS IS THE MATRIX PRODUCT OR DOT PRODUCT

% MATRIX AND ARRAYS
a=randi([0,100],10,10); % make random matrix of intergers 
a(1,10) % index first row, 10th col
a(:,5) % index 5th col
a(1,1:5) % 1st element of the 1st 5 cols


%%Tables

% the function dir (we will see it later) outputs a struct/ table!
listoffiles=dir("./");

load patients
BloodPressure = [Systolic Diastolic];
T = table(Gender,Age,Smoker,BloodPressure);
StructArray = table2struct(T);


%Struct
% used in object oriented programming, useful when you want an object to
% have several features

% Lets make a student struct
student(1).name="Jason";
student(1).age=24;
student(1).GPA= 3.97;
student(1).FavMariahSong= "We belong together";
student(1).buyJoanneoniTunes= true;
student(1).single=true;
student(1).thesisdata=rand(10,10);

%Cells
% are like containers that hold different tupe of info and of different
% sizes!!!!
c={randi([0,100],10,10);randi([0,100],5,5);randi([0,100],2,2); "MARIAH CAREY RULES"};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                        Strings
%           
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

strcmp("Always be my baby", student(1).FavMariahSong)

"Always be my baby"==student(1).FavMariahSong

isItHardtoBelieve=true;
ComeBackBabyPls="pretty please";

song=strcat(" When you left I lost a part of me", int2str(isItHardtoBelieve), ComeBackBabyPls, " because ", student(1).FavMariahSong)

replace(song, "1", ", its so hard to believe ")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                        Loops and Conditionals
%           
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IF CONDITIONAL

a=randi(100,1);

if a< 30
    disp("she's small")
elseif a <70
    disp("she's okay")
else
    disp("SHES HUGE")
end


% Switch
[dayNum, dayString] = weekday(date, 'long', 'en_US');
switch dayString
    case 'Monday'
        disp("UGH Monday")
    case 'Tuesday'
        disp("I'm already tired")
    case 'Wednesday'
        disp("HUMP DAY")
    case 'Thursday'
        disp("I HAVE CLASS TODAY")
    case 'Friday'
        disp("FRIYAY")
    otherwise
        disp('YAS BITCH WERK')
end

% For Loop
for i=1:2:100
    if mod(i,9)==0
        disp(i)
        disp(" is a multiple of 9!!")
    end
end

% While loop
n=1;
nfactorial=1;
while nfactorial <100
    n=n+1;
    nfactorial= nfactorial*n;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                        File I/O
%           
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% useful functions for file navigation:

ls() % list files in your directory
dir() % struct with files of any given directory 
pwd() % tells you where you are currently
cd() % changes your directory 
%   strcat() % covered in strings section, helps you make names of files/dirs
% all these functions operate like in linux with wildcard characters *

% moving directory and loading files !!!!
dir("./justafolderwithdata/")
load("./justafolderwithdata/MLM_demo.mat")
clear all
close all

cd("./justafolderwithdata/")
load('raw_spike_data.mat')
save("temp_data.mat") % saves workspace to file 

%read csv
%   csvread()
%   tdfread()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                        Let's combined what we know
%           
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% move to the right directory, loop through files, select the ones you
% want, open the file, extract the data, and save it!

files= dir('./data_from_sub/sub*.mat');
data_placeholder=[];

for i=1:length(files)
    
    if (mod(str2num( files(i,1).name(5:7) ),2)~=0)
        disp(strcat(files(i,1).name, "   SHES ODD"))
        load(strcat('./data_from_sub/',files(i,1).name));
        data_placeholder(:,:)=data;
        
    end
end

histogram(reshape(data_placeholder,1,size(data_placeholder,1)*size(data_placeholder,2)), 'FaceColor','red')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               PCA
%           We have data collected from the PANAS, a questionaire that rates
%           mood. We hope to find if there are any latent features in this
%           questionaire that we could capture with fMRI.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
% load FC data
load('MLM_demo.mat')

X = categorical(labels);
X = reordercats(X,{'negative','positive','fear','hostility', 'guilt', 'sadness', 'joviality', 'self-assurance', 'attentiveness', 'shyness', 'fatigue', 'serenity', 'surprise'});
bar(X, mean(panasx), 'FaceColor', [1, 0.651, 0.651])
hold on

SEM = std(panasx)/sqrt(length(panasx));               % Standard Error
ts = tinv([0.025  0.975],length(panasx)-1);      % T-Score
CI_low = mean(panasx) - ts(2)*SEM;     % lower 95% CI bound
CI_high = mean(panasx) + ts(2)*SEM;     % Upper 95% CI bound

er = errorbar(X,mean(panasx),CI_low,CI_high);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold off


% NORMALIZE DATA
% subtract mean and divide by stdev, gives you a zscore
%panasx=panasx-mean(panasx);
%panasx=panasx./std(panasx);

% Now compute PCA and visulaize components 
[coeff,score,latent,mu,explained] = pca(panasx); % preform PCA

figure
plot(explained, '-*'); % plot percent var explained
xlabel("Component Number")
ylabel("Percent Variance Explained (%)")

% Now lets reconstitute the data
t=score*coeff' + mean(panasx);

% plot data and compare to original to check that we rotated it back
bar(X, mean(t), 'FaceColor', [1, 0.51, 0.51])
hold on

SEM = std(t)/sqrt(length(t));               % Standard Error
ts = tinv([0.025  0.975],length(t)-1);      % T-Score
CI_low = mean(panasx) - ts(2)*SEM;     % lower 95% CI bound
CI_high = mean(panasx) + ts(2)*SEM;     % Upper 95% CI bound

er = errorbar(X,mean(panasx),CI_low,CI_high);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold off

% plot first two components 
figure
bar(X, coeff(:,1), 'FaceColor', [1, 0.749, 0]) % plot weights of comp one
figure
bar(X, coeff(:,2), 'FaceColor', [0, 0.749, 1]) % plot weights of comp two

% scatter plot how observations fall along the two comp (how are they
% related to one another?)
figure
scatter(score(:,1), score(:,2));

figure
scatter3(score(:,1), score(:,2), score(:, 3));
title('PCA visualization');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               K MEANS
%
%       We have spiking data recorded from an electrode array 
%       from an unknown number of cells recorded in micro volts.
%       We first want to detrend the data, find the spikes/ peaks,
%       and then cluster these peaks with K-Means to find out how many
%       cells we recorded from and which peak belongs to which cell.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
load('raw_spike_data.mat') % load data
sr=44100; % sampling rate of the electrode array
figure
subplot(2,1,1)
plot(raw_spike_data) % plot data to visualize 
axis tight; 
xlabel('time');
ylabel('Micro Volts ');
title('Unfiltered Single Electrode Spike Recording ');

% create and apply filter
% butter requires a ratio of desired cut off frequency and the Nyquist rate
% such that 1= Nyquist rate, thus LOWFREQ/Nyquist; Nyquist rate = 1/2 SR
[b,a] = butter(3,[500/(sr/2) 8000/(sr/2)], 'bandpass'); % create butterworth filter 
filtered_data=filter(b,a,raw_spike_data); % apply butter worth filter 

% plot filtered data with unfiltered data 
subplot(2,1,2)
plot(filtered_data)
axis tight;
xlabel('time');
ylabel('Micro Volts ');
title('Filtered Single Electrode Spike Recording ');

[pks,locs]=findpeaks(-1*filtered_data, 'MinPeakHeight',3.5*std(filtered_data)); % threshold data
% must input the negative version of the data (i.e. -1*) as findpeaks looks
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

% run PCA over peaks for feature extraction; how many components are
% visbale? 
[coeff,score,latent,~,explained] = pca(peaks);
figure
plot(explained, '-*'); % plot percent var explained
xlabel("Component Number")
ylabel("Percent Variance Explained (%)")

% lets plot first two components and see how the data scatters
figure
scatter(score(:,1), score(:,2), '.')

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




% plot 3 cluster solution in PCA space
[ids_opt, crit_opt] = kmeans(score, 2, 'Replicates', 20, 'Options', opts); % 3 cluster optimal solution

figure
scatter(score(ids_opt==1,1), score(ids_opt==1,2), '.g')
hold on;
scatter(score(ids_opt==2,1), score(ids_opt==2,2), '.m')
hold on;
scatter(crit_opt(1:2,1), crit_opt(1:2,2), 200, 'kX')
hold on;
scatter(crit_opt(1:2,1), crit_opt(1:2,2), 200, 'kd')
