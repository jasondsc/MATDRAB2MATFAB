%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               Class 6 
%                              Statistics 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  Descriptive Stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./justafolderwithdata/Iris_2021_data.mat')

mean(iris_data.PetalLength)
median(iris_data.PetalLength)
mode(iris_data.PetalLength)
mean(table2array(iris_data(:,2:7)), 'omitnan') 
mean(table2array(iris_data(:,2:7)), 'all')

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
a(22)=50;
isoutlier(a)
isoutlier(a, 'percentiles', [1 99])
% there are many methods for determining oiutliers 


%%  Correlations and t-tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = randn(8,8);
Y = randn(8,8);

corr(X,Y)
corr2(X,Y)
corrcoef(X,Y)
   
   



%%  Permutations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dating= readtable('./justafolderwithdata/SpeedDatingData.csv');

group1= dating.income(dating.match==0);
group1=group1(~isnan(group1));
group2= dating.income(dating.match==1);
group2=group2(~isnan(group2));
group1=randsample(group1, length(group2));

% compute t

% permute data

% randomly switch 1/2 of the indexes of the two groups

indx_switch=randperm(length(group2),length(group2)/2);
perm1=group1;
perm2=group2;

perm1(indx_switch)=group2(indx_switch);
perm2(indx_switch)=group1(indx_switch);

% compute t again


% loop 


%%  Bootstrapping 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

indx_switch=randi(length(group2),1,length(group2));
perm1=group1(indx_switch);
perm2=group2(indx_switch);

mean(perm1)
mean(perm2)

mean(group1)
mean(group2)

% repeat 

   
%takse function dprime and modify it so that it works along the columns of
%a matrix
% 
%%  Signal Detetction Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


   
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
