%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               Class 6 
%                              Statistics 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
close all

%%  Descriptive Stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./justafolderwithdata/Iris_2021_data.mat')

mean(iris_data.PetalLength)
median(iris_data.PetalLength)
mode(iris_data.PetalLength)
mean(table2array(iris_data(:,2:7)), 'omitnan') 
mean(table2array(iris_data(:,2:7)), 'all')
data=table2array(iris_data(:,2:7));
mean(data(:))

max(table2array(iris_data(:,2:7)))
min(table2array(iris_data(:,2:7)))
maxk(table2array(iris_data(:,2:7)),3)
mink(table2array(iris_data(:,2:7)),3)

std(table2array(iris_data(:,2:7)))
std(table2array(iris_data(:,2:7)),1)
help std
std(table2array(iris_data(:,2:7)),0)
std(table2array(iris_data(:,2:7)),0,2)
std(table2array(iris_data(:,2:7)),1,2)
range(table2array(iris_data(:,2:7)))
var(table2array(iris_data(:,2:7)))

%%  Dealing with missing data and outliers 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a=1:10;
a(2)=NaN

a==NaN
ismissing(a)
rmmissing(a)
a(ismissing(a))=intmin;

a=1:100;
a(22)=500;
isoutlier(a)
a>3*std(a)
isoutlier(a, 'mean')
isoutlier(a, 'percentiles', [1 99])
% there are many methods for determining oiutliers 


% exercise 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load in the speed dating data, remove NaN and compute descriptive stats
% of central tendency and dispersion. Try this without removing NaN, can
% you do it using just the function mean for example? 



%%  Correlations and t-tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = randn(8,8);
Y = randn(8,8);
Z = randn(8,8);

corr(X,Y) % returns the pairwise correlation of columns 
corr2(X,Y) % returns cor coef for vectorized matricies
corrcoef(X,Y) % returns the corr coef for entire 2d matrix
corr2(reshape(X, [1,64]),reshape(Y, [1,64])) % returns cor coef for vectorized matricies

% Let us try a Spearman Rank corr
corr(X,Y, 'Type','Spearman')
corrcoef(X,Y, 'Type','Spearman')
corr2(X,Y, 'Type','Spearman')

corr(X(:), Y(:), 'Type','Spearman')

X = randn(20,3);
Y = randn(20,5);

corr(X,Y) % returns the pairwise correlation of columns 
corr2(X,Y) % returns cor coef for vectorized matricies
corrcoef(X,Y) % returns the corr coef for entire 2d matrix
   
X = randn(10,7);
Y = randn(20,7);

corr(X,Y) % returns the pairwise correlation of columns 
corr2(X,Y) % returns cor coef for vectorized matricies
corrcoef(X,Y) % returns the corr coef for entire 2d matrix


% Reminder:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% corr needs only the same number of rows
% corr2 only takes scalars 
% corrcoef takes 2 matrices of the SAME size
% the function reshpae is very useful when working with data to make sure
% it is the appropriate size 

%%  T-tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% let us read in some data and test if income is an importnat factor in
% speed dating 

dating= readtable('./justafolderwithdata/SpeedDatingData.csv');

group1= dating.income(dating.match==0);
group1=group1(~isnan(group1));
group2= dating.income(dating.match==1);
group2=group2(~isnan(group2));
group1=randsample(group1, length(group2));

clc
help ttest
[h,p,ci,stats]= ttest(group2) % testing the one-sample hypothesis with an alpha of 0.5 and two tails 
% default tests that data come from a normal dist with mean M = 0

% let us change the mean M 
ttest(group2, 4.5e04)


% let us change the tails
ttest(group2, 0, 'Tail', 'left')
ttest(group2, 0, 'Tail', 'right')

ttest(group2, 4.0e04, 'Tail', 'right', 'Alpha', 0.0001)


% paired samples t tests 
[h1,p1,ci1,stats1]= ttest(group1, group2)

[h2,p2,ci2,stats2]= ttest(group1-group2) % same thing as above, WHY?

% independent samples test 
[h2,p2,ci2,stats2]= ttest2(group1, group2)


% effect sizes based on between vs within measures
% see slides for explination of example

% t-value paired
t_paired= (3.75)/(1/sqrt(20)); % with df of 19

% t-value unpaired assuming equal var
sp= ((20-1)*1 + (20-1)*1)/ (20 + 20 -2);
t_unpaired= (3.75)/(sp*sqrt((1/20)+(1/20))); % with df of 38

% Let us half the sample in each group
sp= ((10-1)*1 + (10-1)*1)/ (10 + 10 -2);
t_unpaired= (3.75)/(sp*sqrt((1/10)+(1/10))); % with df of 18


%%  Permutations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dating= readtable('./justafolderwithdata/SpeedDatingData.csv');

group1= dating.income(dating.match==0);
group1=group1(~isnan(group1));
group2= dating.income(dating.match==1);
group2=group2(~isnan(group2));
group1=randsample(group1, length(group2));

% compute t for independent samples, equal var
n=length(group1);
var1=std(group1);
var2=std(group2);
mean1=mean(group1);
mean2=mean(group2);

sp= sqrt(((n-1)*var1^2 + (n-1)*var2^2)/ (n + n -2));
t_unpaired= (mean1-mean2)/(sp*sqrt((1/n)+(1/n))) 
[h,p,ci,stats]=ttest2(group1, group2, 'Vartype', 'equal')


% permute data by randomly switch 1/2 of the indexes of the two groups
% we do this for simplicity, but we do not want to always permute 1/2 the
% data. We want to randomly pick some subset to permute so the second input
% of ranperm needs to change size...


indx_switch=randperm(length(group2),length(group2)/2);
perm1=group1;
perm2=group2;

perm1(indx_switch)=group2(indx_switch);
perm2(indx_switch)=group1(indx_switch);

% compute t again
n=length(perm1);
var1=std(perm1);
var2=std(perm2);
mean1=mean(perm1);
mean2=mean(perm2);

sp= sqrt(((n-1)*var1^2 + (n-1)*var2^2)/ (n + n -2));
t_unpaired= (mean1-mean2)/(sp*sqrt((1/n)+(1/n))) 
[h,p,ci,stats]=ttest2(perm1, perm2, 'Vartype', 'equal')
stats.tstat


% exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create a loop to iterate over 100 permutations of your data. Save the
% result of each permutation (i.e., the resulting t-value) and once done,
% plot the histogram of your null result. What does it look like?
% Use descriptive stats to tell me what your null distribution is like.


%%  Bootstrapping 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bootstrapping usually includes duplicates (i.e., with replacement)...
% Note that permutations usually do NOT include replacement (duplicates)
indx_switch=randi(length(group2),1,length(group2));
perm1=group1(indx_switch);
perm2=group2(indx_switch);

mean(perm1)
mean(perm2)

mean(group1)
mean(group2)

% exercise 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bootstrap your data 100 times and save the mean each time. Now that you
% have 100 estimates of your data plot them and compare it to your actual/
% observed mean. Tell me what you notice.

   
%takse function dprime and modify it so that it works along the columns of
%a matrix
% 
%%  Signal Detetction Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('./dprime/'); % add dprime script to path (modified for this class)

load('./justafolderwithdata/Iris_2021_data.mat');
indx_versicolor=find(strcmp(iris_data.Species, 'versicolor'));
indx_virginica=find(strcmp(iris_data.Species, 'virginica'));
data= iris_data([indx_versicolor,indx_virginica],:);

% plot the data to see where we could draw the line between the two species
figure
histogram(data.PetalLength(1:100))
hold on
histogram(data.PetalLength(101:end))
legend({'versicolor', 'virginica'})

% what is a good decision boundary to take? 

which dprime

% let us use a PetalLength of 4 5 6 as examples
% look at dprime inputs 

% need to compute FA and TP rate
FA= sum(data.PetalLength > 4 & strcmp(data.Species,  'versicolor'))/sum(strcmp(data.Species,  'versicolor'));
HITS= sum(data.PetalLength > 4 & strcmp(data.Species,  'virginica'))/sum(strcmp(data.Species,  'virginica'));

[dpri,ccrit]=dprime(HITS, FA, sum(strcmp(data.Species,  'virginica')),sum(strcmp(data.Species,  'versicolor')))

% how do you interpret these results? What does this mean? Is this good
% classification? 

% lets try it again with 5 and 6
FA= sum(data.PetalLength > 5 & strcmp(data.Species,  'versicolor'))/sum(strcmp(data.Species,  'versicolor'));
HITS= sum(data.PetalLength > 5 & strcmp(data.Species,  'virginica'))/sum(strcmp(data.Species,  'virginica'));

[dpri,ccrit]=dprime(HITS, FA, sum(strcmp(data.Species,  'virginica')),sum(strcmp(data.Species,  'versicolor')))

FA= sum(data.PetalLength > 6 & strcmp(data.Species,  'versicolor'))/sum(strcmp(data.Species,  'versicolor'));
HITS= sum(data.PetalLength > 6 & strcmp(data.Species,  'virginica'))/sum(strcmp(data.Species,  'virginica'));

[dpri,ccrit]=dprime(HITS, FA, sum(strcmp(data.Species,  'virginica')),sum(strcmp(data.Species,  'versicolor')))

% which of the following decision boundaries are the best and why? explain
% yourself 

% exercise 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% code a loop to find optimum of decision boundary 

% take function dprime and modify it so that it works along vectors

[dpri,ccrit]=dprime2d([0 1 0.3], [0.2 0.2 1], [10 10 10], [20 20 20])

%%  ROC curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

count=1;
for i= 10:10:200
x=i;
figure
y = -8:0.1:14;
mu = 2.5;
sigma = 2;
f1 = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot( 0:1:220,f1,'LineWidth',1.5)
hold on
mu = 0;
sigma = 2;
f2 = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot( 0:1:220,f2,'LineWidth',1.5)
xline(x, 'LineWidth', 5)
area(0:1:x-1,f2(:,1:x),'FaceColor', [0.9 0 0],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.1, 'EdgeAlpha', 0)
area(0:1:x-1,f1(:,1:x),'FaceColor', [0.3 0.3 0.8],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.2, 'EdgeAlpha', 0)
area(x-1:1:220,f2(:,x:end),'FaceColor', [0.8 0.3 0.3],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.2, 'EdgeAlpha', 0)
area(x-1:1:220,f1(:,x:end),'FaceColor', [0.0 0.0 0.9],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.1, 'EdgeAlpha', 0)
xlim([0 201])
saveas(gcf, strcat('./ROC_', int2str(count), '.pdf'))
count=count+1;
end


count=1;
mu = 2.5;
sigma = 2;
f1 = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));

mu = 0;
sigma = 2;
f2 = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
figure
for i=(10:10:200)
    
    cri=i;
    if i ==10
        plot(0:0.1:1,0:0.1:1, 'Color', [0.3 0.3 0.3 0.3], 'LineWidth', 5)
        fp1=1;
        tp1=1;
    else
    fp1=fp;
    tp1=tp;
    end
    fp=(10-sum(f2(:,1:i)))/10;
    tp=(10-sum(f1(:,1:i)))/10;
    fptot(count)=fp;
    tptot(count)=tp;

    hold on
    scatter(fp,tp,70,'filled');
    plot([fp1 fp],[tp1 tp],'--', 'Color', [0.3 0.3 0.3 0.3], 'LineWidth', 2)
    %area([fp1 fp],[tp1 tp],'FaceColor', [0.3 0.3 0.3],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.1, 'EdgeAlpha', 0)
    title(' ROC curve analysis')
    xlabel('False Positives')
    ylabel('True Positives')
    saveas(gcf, strcat('./AUC_', int2str(count), '.pdf'))
    count=count+1;
    
end


x=100;
figure
y = -8:0.1:14;
mu = 4.5;
sigma = 2;
f1 = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot( 0:1:220,f1,'LineWidth',1.5)
hold on
mu = 0;
sigma = 2;
f2 = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot( 0:1:220,f2,'LineWidth',1.5)
xline(x, 'LineWidth', 5)
area(0:1:x-1,f2(:,1:x),'FaceColor', [0.9 0 0],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.1, 'EdgeAlpha', 0)
area(0:1:x-1,f1(:,1:x),'FaceColor', [0.3 0.3 0.8],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.2, 'EdgeAlpha', 0)
area(x-1:1:220,f2(:,x:end),'FaceColor', [0.8 0.3 0.3],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.2, 'EdgeAlpha', 0)
area(x-1:1:220,f1(:,x:end),'FaceColor', [0.0 0.0 0.9],'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', 0.1, 'EdgeAlpha', 0)
xlim([0 201])
saveas(gcf, strcat('./ROC_', int2str(93), '.pdf'))


%%  AUC and ROC in MATLAB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./justafolderwithdata/Iris_2021_data.mat')
indx_versicolor=find(strcmp(iris_data.Species, 'versicolor'));
indx_virginica=find(strcmp(iris_data.Species, 'virginica'));
data= iris_data([indx_versicolor,indx_virginica],:);
[X,Y,T,AUC]=perfcurve(data.Species, data.SepalLength, 'virginica')

figure
subplot(1,2,1)
histogram(iris_data.SepalLength(indx_versicolor))
hold on
histogram(iris_data.SepalLength(indx_virginica))
subplot(1,2,2)
plot(X,Y, '-', 'Color', [0.4 0.4 0.4])
hold on
plot(0:0.1:1,0:0.1:1, 'Color', [0.4 0.4 0.4])
scatter(X, Y,10, 'filled', 'MarkerEdgeColor',[0.4 0.4 0.4], 'MarkerFaceColor', [0.4 0.4 0.4] )

% exercise, repeat above steps for all measures and plot all ROC on the
% same curve. Which feature has best AUC (i.e., discrimination) 
