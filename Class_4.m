
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               Class 4 
%                       Plotting graphs in MATLAB
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

%% Reminder on data manipulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data1= rand(10,9);
data2= rand(10,9);
data3= rand(10,9);

hist(rand(1,10000))
hist(randn(1,10000))

data3d=cat(3,data1,data2,data3); % stack 2d matrix into 3d

data3'
data3d' % cannot transpose a 3d matrix

new_data=permute(data3d, [2,3,1]) % reshpaes matrix by reordeing dims

dataall=[data1(:),data2(:),data3(:)]; % use data(:) to access the flattened array

reshape(data3d, [1,2,10,9]) % cannot add a dims
reshape(data3d, [5,2,27])
reshape(data3d, [30,9]) % but you can reduce dims

%%  Descriptive Stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./justafolderwithdata/Iris_2021_data.mat')

mean(iris_data.PetalLength)
median(iris_data.PetalLength)
mode(iris_data.PetalLength)

hist(iris_data.PetalLength)

iris_data2=table2array(iris_data(:,2:7));

iris_data2([20 19 28 17 10 29 18 10], [2 4 6]) = NaN

mean(iris_data2(:,2:6)) 
mean(iris_data2(:,2:6), 'omitnan') 

mean(table2array(iris_data(:,2:7)),2) % along one DIM

mean(reshape(table2array(iris_data(:,2:7)), [1, 6*300] )) 
mean(table2array(iris_data(:,2:7)), 'all') % mean of ALL data 
data=table2array(iris_data(:,2:7));
mean(data(:))

max(table2array(iris_data(:,2:7)),[],2)

max(table2array(iris_data(:,2:7))')

min(table2array(iris_data(:,2:7)))
maxk(table2array(iris_data(:,2:7)),3)
mink(table2array(iris_data(:,2:7)),3)

std(table2array(iris_data(:,2:7)))
std(table2array(iris_data(:,2:7)),1)
help std
std(table2array(iris_data(:,2:7)),[], 2)
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

a(ismissing(a))=mean(a, 'omitnan');

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


% Confidence Intervals 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data2= randn(1,30);

hist(data2)

mean(data2)
std(data2)

for i=1:1000
    data2= randn(1,30);
UCI= mean(data2)+ 1.96* (std(data2)/sqrt(length(data2)))
LCI= mean(data2)- 1.96* (std(data2)/sqrt(length(data2)))


end


% bootstrapping 

data_11= randn(1,30);

id=randperm(length(data_11)); % without replacemnt 

id=randi(length(data_11)); % with replacemnt 

id(1:10)



%% What Not to plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
subplot(1,2,1)
bar([55.9 54.7 55.2 55.3 56 54.4 55.3 54.0])
subplot(1,2,2)
bar([55.9 54.7 55.2 55.3 56 54.4 55.3 54.0])
ylim([50,58]) 
% it is importnat not to over emphasize effects changing the scale of a
% graph may make you effect appear larger than it is. Also remeber to
% always plot a measure of uncertanty or data spread.

 
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
% the colour map you pick and how you center it can over or under emphasize
% effects. Remeber to pick colour bars that are readable for everyone.

usa=readtable('./justafolderwithdata/USA_election_data.csv');

figure
subplot(2,3,1)
plot(usa.Year ,usa.PopularVote)
hold on
plot(usa.Year ,usa.ElectorialCollege)
yline(0.5)

subplot(2,3,2)
bar(usa.Year ,[usa.PopularVote,usa.ElectorialCollege])
hold on
yline(0.5)

subplot(2,3,3)
bar(usa.Year(1:6) ,[usa.PopularVote(1:6),usa.ElectorialCollege(1:6)])
hold on
yline(0.5)

subplot(2,3,4)
bar(usa.Year(7:11) ,[usa.PopularVote(7:11),usa.ElectorialCollege(7:11)])
hold on
yline(0.5)

subplot(2,3,5)
bar(usa.Year(12:end) ,[usa.PopularVote(12:end),usa.ElectorialCollege(12:end)])
hold on
yline(0.5)

subplot(2,3,6)
bar(usa.Year(12:end) ,[usa.PopularVote(12:end),usa.ElectorialCollege(12:end)])
hold on
ylim([0.4 0.7])


%% Plotting basics with plot()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure 
plot(1:1000, 150000-1000*(1:1000)+randi(100000, [1,1000]))
 xlabel('time')
 ylabel('money')
 title('Time is money') % adding labels
 
 data=[150000-1000*(1:1000)+randi(100000, [1,1000]);...
     150000-2000*(1:1000)+randi(100000, [1,1000]);...
     150000-1000*(1:1000)+randi(100000, [1,1000])];
 
 figure
 plot(data')
 
figure 
plot(1:1000, data) % plotting data from multiple groups 
 xlabel('time')
 ylabel('money')
 title('Time is money')
 legend('Karen','Maria', 'Hope') % adding a legend 

 %  specifiers in the plot function can modify the look of the plot
 % you can change the shape, line, and colour easily 
 % also note that Colours can be expressed as [R G B alpha] 
 plot(1:1000, data,'-o','MarkerFaceColor',[.5 .4 .7], 'MarkerEdgeColor',[.7 .7 .7], 'MarkerSize', 10)
% axis([0 100 0 100])
xlim([0 100])

% you can also use functions to redefine the labels of your axes as well as
% their 'ticks'
x = linspace(-10,10,200);
y = cos(x);
figure
subplot(1,2,1)
plot(x,y)
subplot(1,2,2)
plot(x,y)
xticks([-3*pi -2*pi -pi 0 pi 2*pi 3*pi])
xticklabels({'-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi'})
yticks([-1 -0.8 -0.2 0 0.2 0.8 1])
xticklabels({'Load','No Load','Attention','No Attention','Sleepy','dead','alive'})

% plot multiple y axes
figure
yyaxis left
y2 = sin(x/3);
plot(x,y2);
hold on
r = x.^2/2;
yyaxis right
plot(x,r);

yyaxis left
title('Plots with Different y-Scales')
xlabel('Time (S)')
ylabel('Micro Volts')

yyaxis right
ylabel('Power')

 % exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%try plotting three oscilations in one graph and overlay them with
%transparency 


 % Error Bars
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y=rand(100,10);
x=1:10:100;
figure
errorbar(x, mean(y),std(y)) % you can plot data with error bars using the errorbar function



y=rand(100,10);
x=1:10:100;
figure
plot(x, mean(y))
hold on
errorbar(x, mean(y),std(y), 'LineStyle', 'none') % you can plot data with error bars using the errorbar function

y=rand(100,10);
x=1:10:100;
figure
plot(x, mean(y))
hold on
errorbar(x, mean(y),std(y),'CapSize',10, 'LineWidth', 5,  'LineStyle', 'none')

y=rand(100,10);
x=1:10:100;
figure
plot(x, mean(y))
hold on
errorbar(x, mean(y),std(y),'-s','MarkerSize',10,...
    'MarkerEdgeColor','blue','MarkerFaceColor','black','CapSize',20, 'LineWidth', 5)
% The function works exacrly like plot, expect it wants 3 inputs for the
% data. To change the appearnce of the graph see plot above 

figure
plot(x, mean(y))
hold on
errorbar(x, mean(y),std(y),'-s','MarkerSize',10,...
    'MarkerEdgeColor','blue','MarkerFaceColor','black','CapSize',20, 'LineStyle', 'none', 'LineWidth', 5)  
% note how the specifier 'LineStyle' 'none' allows you to control the error
% bars sepeartely from the plot!


figure
scatter(x, mean(y),170,[0.7 0.7 0.7], 'filled')
hold on
scatter(x, mean(y),80,[0.99 0.6 0.7], 'filled')


RPDR_contestant_demo=readtable('./justafolderwithdata/RPDR_contestant_data.csv');
RPDR_episode=readtable('./justafolderwithdata/RPDR_episode_data.csv');
RPDR_contestant=readtable('./justafolderwithdata/RPDR_contestant_score.csv');
RPDR_winners=readtable('./justafolderwithdata/RPDR_winners.csv');

 winnerindex=ismember(RPDR_contestant_demo.contestant, RPDR_winners.contestant(RPDR_winners.winner==1))

 top3index=ismember(RPDR_contestant_demo.contestant, RPDR_winners.contestant(RPDR_winners.winner==0))

 % Bar graphs
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
 

x = categorical({'The Fame' 'Born This Way' 'Artpop' 'Cheek to Cheek' 'Joanne' 'Chromatica'});
x = reordercats(x,{'The Fame' 'Born This Way' 'Artpop' 'Cheek to Cheek' 'Joanne' 'Chromatica'});
data = randi(100, 100, 6);
errhigh =std(data);
errlow  =  std(data);

figure
bar(x,mean(data))                
hold on
er = errorbar(x,mean(data),errlow,errhigh);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

x = linspace(0,25);
y = sin(x/2);


%% Plotting your data's distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Another way to represent you data takes advantaged of the distribution of
% your data. Histograms allow you to visualize your data's shape to make
% conclusions on what effects might be present and what statistical tests you can prefrom etc

 figure
 histogram(RPDR_contestant_demo.age(top3index),6, 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
 hold on;
 histogram(RPDR_contestant_demo.age(winnerindex),6, 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5, 'FaceColor', [0.99 0.5 0.5])
 legend('Top 3', 'Winners')
 xlabel('age')

 % if you know which distribution your data came from or you would like to 
 % compare you data to a distribution you can also plot a probaility density function.
 
 x = 2*randn(5000,1) + 26;
 figure
histogram(x,'Normalization','pdf')
 hold on
y = 10:0.1:40;
mu = 26;
sigma = 2;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi)); % must compute a gaussian or dist yourself
% you can look up the formla for a gaussian
plot(y,f,'LineWidth',1.5)
 
 figure
 histogram(RPDR_contestant_demo.age(top3index),4, 'Normalization','pdf','FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
  hold on
y = 10:0.1:40;
mu = 26;
sigma = 2;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)

 % scatter plots also allow you to visualize the spread of your data
 % very useful to see what is happening across conditions 
iris= load('fisheriris.mat');
iris.meas=iris.meas -mean([iris.meas]);
 
 figure
 scatter(iris.meas(:,1), iris.meas(:,4))
 hold on
 r=corr(iris.meas(:,1), iris.meas(:,4))
 xFit = linspace(min(iris.meas(:,1)), max(iris.meas(:,1)), 1000);
 plot(xFit, r*xFit)
 
 % alternatively you can plot each individual across condition and draw a
 % line between them to see the effect
 
 x=rand(2,100)+0.05;
 figure
 plot(x, '.-','Color', [0.3 0.3 0.3 0.3], 'MarkerSize', 15, 'MarkerFaceColor', [0.3 0.3 0.3 ])
 hold on
 plot(mean(x'), 'o-', 'Color', [0 0 0], 'LineWidth', 5,'MarkerSize', 15, 'MarkerFaceColor', [0 0 1], 'MarkerEdgeColor',[0 0 1] )
 xlim([0 3])
 ylim([ 0 1.1])

%% Other fun plotting tools in MATLAB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% get index of those Queens who won their season and made it to the top 3
 winnerindex=ismember(RPDR_contestant.contestant, RPDR_winners.contestant(RPDR_winners.winner==1))
 top3index=ismember(RPDR_contestant.contestant, RPDR_winners.contestant(RPDR_winners.winner==0))
 
 % lets plot a pie graph with this data
 explode = [1 0 1 0 ]; % determines which section of the graph will stand out
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

% Lets make a word cloud of RuPaul's favoruite musicians
RPDR_episode=RPDR_episode(~ismissing(RPDR_episode.lipsyncartist, 'NA'),:);
lipsynch= groupsummary(RPDR_episode,"lipsyncartist")

figure
wordcloud(lipsynch,'lipsyncartist','GroupCount', 'Color', [0.9 0.5 0.5 ], 'HighlightColor',[0.9 0.3 0.3 ] );
title('Artists to Lipsynch to ')


% staircase plots
X = linspace(0,4*pi,50)';
Y = [0.5*cos(X), 2*cos(X)];

figure
stairs(X,Y)


% Sometimes data, like phases are best represented on a circle rather than
% a regular graph 
Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1000;             % Length of signal
t = (0:L-1)*T;        % Time vector
data = sin(2*pi*2*t).*sin(2*pi*10*t) + rand(1,length(t));
hil=hilbert(data);
figure
subplot(1,3,1)
plot(t, angle(hil))
subplot(1,3,2)
histogram(angle(hil),6)
subplot(1,3,3)
hil=hilbert(data);
polarhistogram(angle(hil),6)

%% Everything is more fun in 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data=randi(100,100,100,100);
figure
scatter3(data(:,1,1), data(1,:,1), data(1,1,:), 'filled')


figure
[X,Y,Z] = sphere(16);
S = repmat([50,25,10],numel(X),1);
C = repmat([1,2,3],numel(X),1);
s = S(:);
c = C(:);
x = [0.5*X(:); 0.75*X(:); X(:)];
y = [0.5*Y(:); 0.75*Y(:); Y(:)];
z = [0.5*Z(:); 0.75*Z(:); Z(:)];
scatter3(x,y,z,s,c, 'filled')

 

%% Plotting with the Gramm Toolbox
% % I'm not a regular graph, I am a cool graph
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gramm is a toolbox that can be downloaed to extend MATLAB's graphics
% capacities. Gramm works with a similar syntax to ggplot in R and as such
% is significantly different than MATLAB's syntax. Below are a few examples
% of different plots you can make with gramm

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                              WARNING YOU NEED TO ADD
%                             GRAMM TO YOUR PATH TO RUN
%                                 THE FOLLOWING CODE

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('~/Documents/gramm-master/')
load example_data.mat

g=gramm('x',cars.Model_Year,'y',cars.MPG,'color',cars.Cylinders,'subset',cars.Cylinders~=3 & cars.Cylinders~=5);
g.facet_grid([],cars.Origin_Region);
g.geom_point();
g.stat_glm();
g.set_names('column','Origin','x','Year of production','y','Fuel economy (MPG)','color','# Cylinders');
g.set_title('Fuel economy of new cars between 1970 and 1982');
figure('Position',[100 100 800 400]);
g.draw();


x=randn(1200,2);
c=cell(length(x),1);
c(:)={'JLo'};

c2=cell(length(x),1);
c2(:)={'Mariah Carey'};
cond={c{:};c2{:}};
x(:,2)=x(:,2)+10;

clear g5
g5(1,1)=gramm('x',x','color',cond);
g5(1,2)=copy(g5(1));
g5(1,3)=copy(g5(1));
g5(1,1).stat_bin('geom','stacked_bar'); %Stacked bars option
g5(1,1).set_title('''stacked_bar''');
g5(1,2).stat_bin('geom','overlaid_bar'); %Overlaid bar automatically changes bar coloring to transparent
g5(1,2).set_title('''overlaid_bar''');
g5(1,3).stat_bin('geom','stairs'); %Default fill is edges
g5(1,3).set_title('''stairs''');
g5.set_title('''geom'' options for stat_bin()');
g5.set_names('x','Number #1 Singles');
figure('Position',[100 100 800 600]);
g5.draw();



clear g
g(1,1)=gramm('x',cars.Origin_Region,'y',cars.Horsepower,'color',cars.Cylinders,'subset',cars.Cylinders~=3 & cars.Cylinders~=5);
g(1,2)=copy(g(1));
%Averages with confidence interval
g(1,1).stat_summary('geom',{'bar','black_errorbar'});
g(1,1).set_title('stat_summary()');
%Boxplots
g(1,2).stat_boxplot();
g(1,2).set_title('stat_boxplot()');
%These functions can be called on arrays of gramm objects
g.set_names('x','Origin','y','Horsepower','color','# Cyl');
g.set_title('Visualization of Y~X relationships with X as categorical variable');
figure('Position',[100 100 800 550]);
g.draw();


% simulate oscilatory data with sin
fs = 1000; % Sampling frequency (samples per second) 
dt = 1/fs; % seconds per sample 
StopTime = 2; % seconds 
time = (0:dt:StopTime)'; % seconds 
F = 60; % Sine wave frequency (hertz) 
osci1 = sin(2*pi*5*time)+ sin(2*pi*10*time);
osci2 = sin(2*pi*2*time) + sin(2*pi*120*time);
osci3 = sin(2*pi*F*time)+ sin(2*pi*102*time);

clear g
figure
g(1,1)=gramm('x',repmat(time,[1,3])','y', [osci1, osci2, osci3]', 'color',{'Osci1', 'Osci2', 'Osci3'});
%smooth plot 
g(1,1).geom_line();
g(1,1).set_title('Periodic --Baseline');
g(1,1).set_color_options('map','brewer2');
%These functions can be called on arrays of gramm objects
g.set_names('x','Frequency (Hz)','y','Log Power');
g.draw();

clear g
figure
g(1,1)=gramm('x',repmat(time,[1,3])','y', [osci1, osci2, osci3]', 'color',{'Osci1', 'Osci2', 'Osci3'});
%smooth plot 
g(1,1).geom_line();
g.facet_grid([], {'Osci1', 'Osci2', 'Osci3'}');
g(1,1).set_title('Periodic --Baseline');
g(1,1).set_color_options('map','brewer2');
%These functions can be called on arrays of gramm objects
g.set_names('x','Frequency (Hz)','y','Log Power');
%g.axe_property('YLim',[1e-15 1.5e-9]);
%g.axe_property('XLim',[0 50]);
g.draw();




% exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot two normal distributions and move a verticle line (i.e., a
% criterion) across all integer x values 




