

 
 %%%%%%%%%%%%%%%%%%%%%%%%%%
%% class 5

clear all 
close all

figure
subplot(1,2,1)
bar([55.9 54.7 55.2 55.3 56 54.4 55.3 54.0])
subplot(1,2,2)
bar([55.9 54.7 55.2 55.3 56 54.4 55.3 54.0])
ylim([50,58])
 

image=imread('./MemeFolder/ladygaga4.jpg');


image=imgaussfilt(mean(double(image),3),100);
 
figure
subplot(2,2,1)
imagesc(image)
colorbar
colormap(gca, jet)
subplot(2,2,2)
imagesc(image)
colorbar
colormap(gca, pink)
set(gca, 'YDir','reverse')
subplot(2,2,3)
imagesc(image)
colorbar
axis off;
colormap(gca, colorcube)
subplot(2,2,4)
imagesc(image)
colorbar
colormap(gca, jet)
caxis([50 200])


figure 
plot(1:1000, 150000-1000*(1:1000)+randi(100000, [1,1000]))
 xlabel('time')
 ylabel('money')
 title('Time is money')
 
 data=[150000-1000*(1:1000)+randi(100000, [1,1000]);...
     150000-2000*(1:1000)+randi(100000, [1,1000]);...
     150000-1000*(1:1000)+randi(100000, [1,1000])];
 
figure 
plot(1:1000, data)
 xlabel('time')
 ylabel('money')
 title('Time is money')
 legend('Karen','Maria', 'Hope')
 
 
 plot(1:1000, data,'-o','MarkerFaceColor',[.5 .4 .7], 'MarkerEdgeColor',[.7 .7 .7], 'MarkerSize', 10)
% axis([0 100 0 100])
xlim([0 100])

RPDR_episode=readtable('./justafolderwithdata/RPDR_episode_data.csv');
 RPDR_contestant=readtable('./justafolderwithdata/RPDR_contestant_score.csv');
 RPDR_winners=readtable('./justafolderwithdata/RPDR_winners.csv');

 winnerindex=ismember(RPDR_contestant.contestant, RPDR_winners.contestant(RPDR_winners.winner==1))

 top3index=ismember(RPDR_contestant.contestant, RPDR_winners.contestant(RPDR_winners.winner==0))
 
 explode = [1 0 1 0 ];
labels = {'Wins','High','Low', 'Btm'};
figure
subplot(1,2,1)
pie([sum(RPDR_contestant.WIN(winnerindex)), sum(RPDR_contestant.HIGH(winnerindex)),...
    sum(RPDR_contestant.LOW(winnerindex)), sum(RPDR_contestant.BTM(winnerindex))], explode, labels)
 title(gca, 'Winner of their season', 'Position',[-0.12 -1.5 0])
 colormap(gca, cool)
 
 subplot(1,2,2)
pie([sum(RPDR_contestant.WIN(top3index)), sum(RPDR_contestant.HIGH(top3index)),...
    sum(RPDR_contestant.LOW(top3index)), sum(RPDR_contestant.BTM(top3index))], explode, labels)
 title(gca, 'Top 3', 'Position',[0.02, -1.5 0])
colormap(gca, cool)
 
RPDR_episode=RPDR_episode(~ismissing(RPDR_episode.lipsyncartist, 'NA'),:);
lipsynch= groupsummary(RPDR_episode,"lipsyncartist")

figure
wordcloud(lipsynch,'lipsyncartist','GroupCount');
title("Artists to LipSynch to")

 
 RPDR_contestant_demo=readtable('./justafolderwithdata/RPDR_contestant_data.csv');
 
 
 winnerindex=ismember(RPDR_contestant_demo.contestant, RPDR_winners.contestant(RPDR_winners.winner==1))

 top3index=ismember(RPDR_contestant_demo.contestant, RPDR_winners.contestant(RPDR_winners.winner==0))

 figure
 histogram(RPDR_contestant_demo.age(top3index),10, 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
 hold on;
 histogram(RPDR_contestant_demo.age(winnerindex),10, 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5, 'FaceColor', [0.99 0.5 0.5])
 legend('Top 3', 'Winners')
 xlabel('age')
 
 
 
 
 % scatter 
iris= load('fisheriris.mat');
iris.meas=iris.meas -mean([iris.meas]);
 
 figure
 scatter(iris.meas(:,1), iris.meas(:,4))
 hold on
 r=corr(iris.meas(:,1), iris.meas(:,4))
 xFit = linspace(min(iris.meas(:,1)), max(iris.meas(:,1)), 1000);
 plot(xFit, r*xFit)
 
 
 
 data=[sum(RPDR_contestant.WIN(winnerindex)), sum(RPDR_contestant.HIGH(winnerindex)),...
    sum(RPDR_contestant.LOW(winnerindex)), sum(RPDR_contestant.BTM(winnerindex)); ...
    sum(RPDR_contestant.WIN(top3index)), sum(RPDR_contestant.HIGH(top3index)),...
    sum(RPDR_contestant.LOW(top3index)), sum(RPDR_contestant.BTM(top3index))];

figure
bar(data)
legend('Wins','High','Low', 'Btm')
figure
bar(data, 'stacked')
legend('Wins','High','Low', 'Btm')

X = categorical({'Monday','Tuesday','Wednesday','Thursday', 'Friday'});
Y = [54 44 49 29 10];
bar(X,Y)
X = reordercats(X,{'Monday','Tuesday','Wednesday','Thursday', 'Friday'});
bar(X,Y)
yticks([0  20  40  60 ])
 
 
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
   
   
