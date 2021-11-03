
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               Class 6
%                     Data Reduction and Clustering
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

% Below we will apply PCA and ICA to a variety of data sets to observe what
% happens when we run each algorithm. We will use several examples from
% single images, a dataset of images, to sounds.

%%  PCA on a single image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here we will observe what happens when you run PCA on a single image

image=imread('./MemeFolder/WhoAmI.jpg'); % read in image to MATLAB
%image=imread('./MemeFolder/ladygaga4.jpg');
image=mean(double(image),3); % forcing the input to be a double and 
% then taking a mean along the RGB values makes the image black and white 

% run PCA
[coeff, score, latent, tsquared, explained, mu]=pca(image);
% coeff --matrix (n by n) of weights, each PC is a col
% score --matrix (m by n) describes how each observation loads on a PC
% latent --eigenvalues of the covariance matrix of image
% tsquared --- Sum of squares of standardized scores
% explained -- variance explained by each component 
% mu -- the estimated means of the columns of your data

figure
plot(explained, '-o', 'MarkerSize',10)
xlim([1 20])

% Lets visulaize the output 
figure
subplot(3,3,9)
imagesc(image)
colormap(gray)
axis off;

% Let's multiply the scores by the coeff to see what our image would look
% like with just 1, 2, 5, 10 etc components 
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
% notice what happens when we add more components to the image

% exercise 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% why is it that the last few components we add do very little to improve
% the image 

% Try scripting the above plots so that the code is more elegant


%%  ICA on a single image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here we will observe what happens when you run ICA on a single image

% plot original image below
figure
subplot(3,4,12)
imagesc(image)
colormap(gray)
axis off;

ica_out=rica(image,50); % run ICA algorithm on data 
% retruns struct with TransformWeights
t=transform(ica_out,image); % transforms/ rotates the original data into new space
% t is much like the score you get in PCA

% Let's do the same thing, and add several components to the data
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

% exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% what do you notice that is different than PCA? 
% How does adding additional compoents change the overall picture?
% Does each component carry the same amount of information or weight?
% What does each component look like?

%%  PCA and ICA on a group of picture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here we will observe what happens when you run PCA  and ICA on a group
% of pictures (i.e., a database of images all of faces).

clear all
close all

files= dir('./justafolderwithdata/ServingUFace/*.png'); % get list of images
count=1;
% iterate through files and load each one
for i=1:length(files)
    
    temp=imread(strcat('./justafolderwithdata/ServingUFace/', files(i,1).name));
    temp=imresize(temp,0.05); % downsample image to run faster
    temp=mean(double(temp),3); % cast as double and make image greyscale
    data(count,:,:)=temp;
    count=count+1;
end

% plot mean of all images 
figure
subplot(3,3,1)
imagesc(squeeze(mean(data,1)))
colormap(gray)
axis off;
s=size(data,3);
% reshape data so that rows are observations (each individual image) and
% cols are features (i.e., each individual pixel of image) 
image=reshape(data, [69, s*s]);
% run PCA
[coeff, score, latent, tsquared, explained, mu]=pca(image);
subplot(3,3,2)
% plot each component of PCA to see what features they found
imagesc(squeeze(reshape(coeff(:,1),[1, s,s])))
colormap(gray)
axis off;
subplot(3,3,3)
imagesc(squeeze(reshape(coeff(:,2),[1, s,s])))
colormap(gray)
axis off;
subplot(3,3,4)
imagesc(squeeze(reshape(coeff(:,3),[1, s,s])))
colormap(gray)
axis off;
subplot(3,3,5)
imagesc(squeeze(reshape(coeff(:,4),[1, s,s])))
colormap(gray)
axis off;
subplot(3,3,6)
imagesc(squeeze(reshape(coeff(:,5),[1, s,s])))
colormap(gray)
axis off;
subplot(3,3,7)
imagesc(squeeze(reshape(coeff(:,6),[1, s,s])))
colormap(gray)
axis off;
subplot(3,3,8)
imagesc(squeeze(reshape(coeff(:,7),[1, s,s])))
colormap(gray)
axis off;
subplot(3,3,9)
imagesc(squeeze(reshape(coeff(:,8),[1, s,s])))
colormap(gray)
axis off;
% what do you notice about the features PCA extracts?
% open up some of the images in the directroy and compare them to your
% componets.
% what are the similarities and differences?

% lets try ICA on the same data
tic
ica_out=rica(image,length(files), 'Standardize', true);
toc % find out how long this command takes to run

% plot mean of images again
figure
subplot(3,4,12)
imagesc(squeeze(mean(data,1)))
colormap(gray)
axis off;

% plot individual extracted components from ICAS
subplot(3,4,1)
imagesc(reshape(ica_out.TransformWeights(:,1)', [s,s]))
colormap(gray)
axis off

subplot(3,4,2)
imagesc(reshape(ica_out.TransformWeights(:,2)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,3)
imagesc(reshape(ica_out.TransformWeights(:,3)', [s,s]))
colormap(gray)
axis off;


subplot(3,4,4)
imagesc(reshape(ica_out.TransformWeights(:,4)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,5)
imagesc(reshape(ica_out.TransformWeights(:,5)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,6)
imagesc(reshape(ica_out.TransformWeights(:,6)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,7)
imagesc(reshape(ica_out.TransformWeights(:,7)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,8)
imagesc(reshape(ica_out.TransformWeights(:,8)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,9)
imagesc(reshape(ica_out.TransformWeights(:,9)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,10)
imagesc(reshape(ica_out.TransformWeights(:,10)', [s,s]))
colormap(gray)
axis off;

subplot(3,4,11)
imagesc(reshape(ica_out.TransformWeights(:,11)', [s,s]))
colormap(gray)
axis off;

% exercise 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% what sorts of features does ICA extract?
% Compare these to the original images? Plot a downsampled, greyscale image
% to compare
% How is this different than PCA?



%%  Blind Source Seperation Problem (ICA and PCA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng default % set the random number generator of MATLAB

% let us load a bunch of sounds into matlab
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
% let us mix up the sounds into complicated sounds
% as if these were all happening in the same room and we recorded them with
% different microphones
mixdata = S*randn(6) + randn(1,6);

% let us plot the original sounds and the mixed up signals
figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(mixdata(:,i))
    title(['Mix ',num2str(i)])
end

% Lets play an example of the original sound and the mixed sound
soundsc(S(:,1));
pause(2)
soundsc(mixdata(:,1))

% Let's run PCA
[coeff, score, latent, tsquared, explained, mu]=pca(mixdata);

figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(score(:,i))
    title(['Unmix ',num2str(i)])
end

% play unmixed sounds
soundsc(S(:,1));
pause(2)
soundsc(score(:,1));

% exercise 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How well did PCA do?
% Why do you think this did or did not work?
% what could the potential problems here?


% How about ICA? 
mixdata = prewhiten(mixdata);
q = 6;
Mdl = rica(mixdata,q,'NonGaussianityIndicator',ones(6,1));

unmixed = transform(Mdl,mixdata);

% plot unmixed sounds
figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(unmixed(:,i))
    title(['Unmix ',num2str(i)])
end

% play unmixed sounds
soundsc(S(:,1));
pause(2)
soundsc(unmixed(:,2))

% exercise 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How well did ICA do?
% Why do you think this did or did not work?


%%    K- means
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Let's take up our example of spikes recorded from an electrode array 
% see previous classes for explination of filters 
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
opts = statset('Display','final'); % set options of k-means
numclust=10; % max number of k

% make place holder for output 
ctrs = zeros(size(peaks,1), numclust);
ids = zeros(size(peaks,1), numclust);
sumd= zeros(1,numclust);

%iterate through different k and run k-means
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
scatter(score(ids_opt==1,1), score(ids_opt==1,2), 20, [0.4, 0.9 0.4], 'filled')
hold on;
scatter(score(ids_opt==2,1), score(ids_opt==2,2), 20, [0.4, 0.4 0.9], 'filled')
hold on;
scatter(score(ids_opt==3,1), score(ids_opt==3,2), 20, [0.9, 0.4 0.4], 'filled')
hold on;
scatter(crit_opt(1:3,1), crit_opt(1:3,2), 200, 'kX')
hold on;
scatter(crit_opt(1:3,1), crit_opt(1:3,2), 200, 'kd')



%%    K- means example of behavioural data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dataset from a speedating experiment where information about
% both parties is collected

% load data
dating=readtable('./justafolderwithdata/modified_lovoo_v3_users_instances.csv');

data=dating(:,[2:8 10:end-1]); % remove unwanted columns

data=normalize(table2array(data)); % normalize data before clustering 

% run PCA over data
[coeff,score,latent,~,explained] = pca(data);
figure
plot(explained, '-*'); % plot percent var explained
xlabel("Component Number")
ylabel("Percent Variance Explained (%)")

% lets plot first two components and see how the data scatters
figure
scatter(score(:,1), score(:,2), '.')

% get subset of data to act as training and testing sets 
[train, indx]= datasample(data, floor(2.*size(data,1)./3),'Replace', false);
test= data(setdiff(1:length(data), indx),:);

% Run K-means clustering across 1-10 ks
opts = statset('Display','final');
numclust=10; % max number of k
ctrs = zeros(size(train,1), numclust);
ids = zeros(size(train,1), numclust);
sumd= zeros(1,numclust);

for j=1:numclust
[ids(:,j)] = kmeans(train, j, 'Replicates', 20, 'Options', opts);

end

% find the optimal solution for the best number of clusters
eva_cali = evalclusters(train,ids,'CalinskiHarabasz') % evaluate the best number of k clusters
eva_sil = evalclusters(train,ids,'silhouette') % evaluate the best number of k clusters
eva_Dav = evalclusters(train,ids,'DaviesBouldin') % evaluate the best number of clusters

% all three methods returned different numbers, plot the silhouettes
figure
silhouette(train,ids(:,2))
figure
silhouette(train,ids(:,3))
figure
silhouette(train,ids(:,7))

score_test=score(setdiff(1:length(data), indx),:);
% plot 3 cluster solution in PCA space
[ids_opt, crit_opt] = kmeans(score_test, 2, 'Replicates', 20, 'Options', opts); % 3 cluster optimal solution

figure
scatter(score_test(ids_opt==1,1), score_test(ids_opt==1,2), 20, [0.4, 0.9 0.4], 'filled')
hold on;
scatter(score_test(ids_opt==2,1), score_test(ids_opt==2,2), 20, [0.4, 0.4 0.9], 'filled')
scatter(crit_opt(1:2,1), crit_opt(1:2,2), 200, 'kX')
scatter(crit_opt(1:2,1), crit_opt(1:2,2), 200, 'kd')

score_test=score(setdiff(1:length(data), indx),:);
% plot 2 cluster solution in PCA space
[ids_opt, crit_opt] = kmeans(score_test, 2, 'Replicates', 20, 'Options', opts); % 3 cluster optimal solution

figure
scatter(score_test(ids_opt==1,1), score_test(ids_opt==1,2), 20, [0.4, 0.9 0.4], 'filled')
hold on;
scatter(score_test(ids_opt==2,1), score_test(ids_opt==2,2), 20, [0.4, 0.4 0.9], 'filled')
scatter(crit_opt(1:2,1), crit_opt(1:2,2), 200, 'kX')
scatter(crit_opt(1:2,1), crit_opt(1:2,2), 200, 'kd')

% now that we classified the daters into groups, lets figure out if there
% are any differences between these categories 
data_explore=dating(:,[2:8 10:end-1]);
data_explore=data_explore(indx, :);

data_1=data_explore(ids(:,3)==1, :);

data_2=data_explore(ids(:,3)==2, :);

data_3=data_explore(ids(:,3)==3, :);

[mean(data_1.age) mean(data_2.age)]

[mean(data_1.counts_profileVisits) mean(data_2.counts_profileVisits)]

[mean(data_1.counts_pictures) mean(data_2.counts_pictures) ]

[mean(data_1.lang_count) mean(data_2.lang_count)]

[mean(data_1.isVerified) mean(data_2.isVerified)]

% In sum add lost of pictures to your dating profile, get verified, and
% learn a new langugae (Does MATLAB count?)

