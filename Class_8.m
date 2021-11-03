
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               Class 8 
%                     Machine Learning: Regression & SVM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

% Below I will walk you through several examples of regressions and the
% different types of models you can build. Regressions are powerful tools
% and take on many different forms. Please see the slides for the theory

%%  Linear Regression Models 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Let us try and find out what predicts how many main challenge wins a Drag
% queen gets on RuPaul's Drag Race (not sponsored) using Linear Regression

% Load Drag Race data
RPDR_contestant_demo=readtable('./justafolderwithdata/RPDR_contestant_data.csv');
RPDR_episode=readtable('./justafolderwithdata/RPDR_episode_data.csv');
RPDR_contestant=readtable('./justafolderwithdata/RPDR_contestant_score.csv');
RPDR_winners=readtable('./justafolderwithdata/RPDR_winners.csv');

% get index of each contestant in the other matrix to collapse into one
% large data matrix
[indexA, indexB]=ismember(RPDR_winners.contestant,RPDR_contestant.contestant);

[indexC, indexD]=ismember(RPDR_winners.contestant,RPDR_contestant_demo.contestant);

% table all the variables we wish to explore into one large data frame
data=table(RPDR_winners.contestant, [RPDR_contestant_demo.age(indexD)],...
     table2array(RPDR_contestant(indexB, 3)), table2array(RPDR_contestant(indexB, 4)),...
      table2array(RPDR_contestant(indexB, 5)), table2array(RPDR_contestant(indexB, 6)),...
       table2array(RPDR_contestant(indexB, 7)),RPDR_winners.winner);
data.Properties.VariableNames = {'contestant' 'age' 'BTM' 'LOW', 'SAFE','HIGH', 'WIN', 'winner'} % rename columns

% now that we have rearranged our data and put it in a format that we can
% work with (i.e., long format) let us run a regression model 
X = table(ones(size(data.age)), data.age, data.BTM, data.LOW, data.SAFE);

[b,bint,r,rint,stats] = regress(data.WIN,table2array(X))    % Removes NaN data
% b will return the beta coefficents
% bint will retrun the 95% CI of the betas
% r returns the residulas
% rint diagnostic intervals to check for outliers 
% stats returns the R2, the F, the p-value, and the estimated error
% variance

% Let us plot one regression model
% First compute the linear fit.
X = table(ones(size(data.age)),  data.BTM);
[b,bint,r,rint,stats] = regress(data.WIN,table2array(X))    % Removes NaN data
Slope = b(2)
Intercept = b(1)
% Plot training data and fitted data.
aFitted = data.BTM; % Evalutate the fit as the same x coordinates.
bFitted = Intercept + aFitted.*Slope;
figure
scatter(data.BTM, data.WIN, 20, 'MarkerEdgeColor',[0.9 0.4 0.4 ] ,'MarkerFaceColor',[0.9 0.4 0.4 ]) ;
hold on;
plot(aFitted, bFitted, '*-','Color', [0.4 0.4 0.9], 'LineWidth', 2);
grid on;
xlabel('Wins', 'FontSize', 20);
ylabel('Btm2', 'FontSize', 20);

% loop over x to draw a line between observed and fitted data
for k = 1 : length(aFitted)
  yActual = data.WIN(k);
  yFit = bFitted(k);
  x = data.BTM(k);
  plot([x, x], [yFit, yActual], '--','Color', 'm');
end
% this essentially plots the residuals (see below) 

% Let us try regressing another example
dating=readtable('./justafolderwithdata/modified_lovoo_v3_users_instances.csv');

dating=dating(:,[2:8 10:end-1]);
X = table(ones(size(dating.age)), dating.age);
[b,bint,r,rint,stats] = regress(dating.counts_profileVisits,table2array(X))    % Removes NaN data

% plot regression fit
% First compute the linear fit.
Slope = b(2)
Intercept = b(1)
% Plot training data and fitted data.
aFitted = dating.age; % Evalutate the fit as the same x coordinates.
bFitted = Intercept + aFitted.*Slope;
figure
scatter(dating.age, dating.counts_profileVisits, 20, 'MarkerEdgeColor',[0.9 0.4 0.4 ] ,'MarkerFaceColor',[0.9 0.4 0.4 ]) ;
hold on;
plot(aFitted, bFitted, '*-','Color', [0.4 0.4 0.9], 'LineWidth', 2);
grid on;
xlabel('Age', 'FontSize', 20);
ylabel('Profile Visits', 'FontSize', 20);
ylim([0 10000])

for k = 1 : length(aFitted)
  yActual = dating.counts_profileVisits(k);
  yFit = bFitted(k);
  x = dating.age(k);
  plot([x, x], [yFit, yActual], '--','Color', 'm');
end

%%  Nuissance variables and Residulas 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nuissance variables are variables we do not care about and want to remove
% their effect from our data

% Let us imagine that we think age is a confound in online dating, so we
% wish to remove the effect of age on profile visits 

% Regress out effect 
X = table(ones(size(dating.age)), dating.age);
[b,bint,r,rint,stats] = regress(dating.counts_profileVisits,table2array(X))    % Removes NaN data
Slope = b(2)
Intercept = b(1)
aFitted = dating.age; % Evalutate the fit as the same x coordinates.
bFitted = Intercept + aFitted.*Slope;
% compute residuals as y minus slope of effect (leave the intercept alone
% for now)
residulas = dating.counts_profileVisits - aFitted.*Slope;

% fit a line between age and the residuals 
X = table(ones(size(dating.age)), dating.age);
[b,bint,r,rint,stats] = regress(residulas,table2array(X))    % Removes NaN data
Slope2 = b(2)
Intercept2 = b(1)
aFitted2 = dating.age; % Evalutate the fit as the same x coordinates.
bFitted2 = Intercept2 + aFitted.*Slope2;

% plot the original effect and the outcome after removing the effect of age
figure
subplot(1,2,1)
histogram(dating.counts_profileVisits)
subplot(1,2,2)
histogram(residulas)

figure
subplot(1,2,1)
scatter(dating.age,dating.counts_profileVisits)
hold on
plot(aFitted, bFitted)
xlabel('Age', 'FontSize', 20);
ylabel('Profile Visits', 'FontSize', 20);
ylim([0 30000])
subplot(1,2,2)
scatter(dating.age,residulas)
hold on
plot(aFitted2, bFitted2)
xlabel('Age', 'FontSize', 20);
ylabel('Profile Visits', 'FontSize', 20);
ylim([0 30000])


% you can also fit linear regressions with the function fitlm
% See MATLAB's webpage for more details about differences 
X = table( dating.age, dating.counts_profileVisits);
fit=fitlm(X);
fit.Residuals % retruns table with different residulas 
% Raw represents observed y value - predicted 

% exercise 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% try running the above code where we remove the effect of age but also
% remove the intercept. What happens? What do the residulas look like now?
% Compare these to the r. What is the Intercept based on these results? 
% Why would you want to remove the interceot? When
% Would it be a good idea to keep the intercept?


%%  Effects of Coding Scheme on Regressions and Interpretations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% below we will test the effects of different coding schemes on regression
% models and their results. We will discuss in class the consequences and
% the interpretation of each scheme 

% dummy coded effect (0,1)
X = table(ones(size(data.age)), data.age, data.winner);
[b,bint,r,rint,stats] = regress(data.WIN,table2array(X))    % Removes NaN data

% let us try effect coded (-1,1)
data.winner_effectcode=data.winner;
data.winner_effectcode(data.winner==0)=-1
X = table(ones(size(data.age)), data.age, data.winner_effectcode);
[b2,bint2,r,rint,stats] = regress(data.WIN,table2array(X)) 

% how are the beta estimates different? 

% Let us try effect coding with -0.5 and 0.5
data.winner_effectcode=data.winner;
data.winner_effectcode(data.winner==0)=-0.5;
data.winner_effectcode(data.winner==1)=0.5;

X = table(ones(size(data.age)), data.age, data.winner_effectcode);
[b3,bint3,r,rint,stats] = regress(data.WIN,table2array(X)) 

% let us compare all three betas and their 95% CI
betas=[b b2 b3]
betas_int=table(bint, bint2,  bint3)


% exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% try making up your own coding schme (e.g., 1 2) how does this change the value of the
% beta and its interpretation. What does that intercept in your model
% reflect now?


%%  Logistic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sometimes our outcome variables are not continuous, in this case fitting
% a line just wont cut it. See slides for examples and theory

% example of logistic regression same interpretation as regress
X = [ones(size(data.age)), data.age, data.WIN];
[B,dev,stats] = mnrfit(X,categorical(data.winner));

% Let us compare what a linear vs logistic regression on a binary variable
% looks like
load fisheriris

spplot=[repmat(0,50,1);repmat(1,50,1)];
measplot=meas(1:100,1);

[b,bint,r,rint,stats] = regress(spplot,[repmat(1,100,1) measplot]);
figure
scatter(measplot, spplot, 50,'filled','MarkerFaceColor', [0.9 0.4 0.4])
hold on
x=3:0.05:7;
y=b(1) +b(2)*x;
plot(x,y, '--','LineWidth', 3, 'Color', [0.7 0.7 0.7])

[B,dev,stats] = mnrfit(measplot,categorical(spplot));
figure
scatter(measplot, spplot, 50,'filled','MarkerFaceColor', [0.9 0.4 0.4])
hold on
x=3:0.05:7;
y= 1./(1+exp((B(1) +B(2)*x)));
plot(x,y, '--','LineWidth', 3, 'Color', [0.7 0.7 0.7])

%%  Hierarchical Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% When the data you are looking at is nested, i.e., has a correlational
% structure, regression models should take this into account to reduce
% noise and get better estimates of your effects
% see http://mfviz.com/hierarchical-models/ and slide for details
load('./justafolderwithdata/Iris_2021_data.mat');

% Run linear regression on iris data plot across species
figure
subplot(1,2,1)
scatter(iris_data.SepalLength, iris_data.SepalWidth, 20, categorical(iris_data.Species), 'filled')
hold on
X = table(ones(size(iris_data.SepalWidth)), iris_data.SepalLength);
[b,bint,r,rint,stats] = regress(iris_data.SepalWidth,table2array(X))    % Removes NaN data
Slope = b(2)
Intercept = b(1)
% Plot training data and fitted data.
aFitted = iris_data.SepalLength; % Evalutate the fit as the same x coordinates.
bFitted = Intercept + aFitted.*Slope;
plot(aFitted, bFitted, '--', 'LineWidth', 3, 'Color', [0.4 0.4 0.4])
xlabel('Sepal Length', 'FontSize', 20);
ylabel('Sepal Width', 'FontSize', 20);

% as you can notice the line fit across species may represent the best
% fitting line across ALL data but this does not reflect the actual trends
% within each species. A much better model would allow the fitted line to
% vary as a function of species 

% get index of each species, this will be useful later 
indx_setosa=find(strcmp(iris_data.Species, 'setosa'));
indx_versicolor=find(strcmp(iris_data.Species, 'versicolor'));
indx_virginica=find(strcmp(iris_data.Species, 'virginica'));

% run a hierarchical model where each species is given a different
% intercpet. See slides for details 

% Note that variables outside the () will be FIXED and those inside will be
% RANDOM, this will be explained in class
formula         = 'SepalWidth ~ SepalLength + (1|Species)';
tbl             = table(iris_data.SepalLength,iris_data.SepalWidth,iris_data.Species,  'VariableNames',{'SepalLength','SepalWidth','Species'});
lme             = fitlme(tbl,formula);

% estimate the data for all species 
rand_eff=lme.randomEffects;
aFitted1= iris_data.SepalLength(indx_setosa);
bFitted1= lme.Coefficients.Estimate(1) + iris_data.SepalLength(indx_setosa).*lme.Coefficients.Estimate(2) + rand_eff(1);

aFitted2= iris_data.SepalLength(indx_versicolor);
bFitted2= lme.Coefficients.Estimate(1) + iris_data.SepalLength(indx_versicolor).*lme.Coefficients.Estimate(2) + rand_eff(2);

aFitted3= iris_data.SepalLength(indx_virginica);
bFitted3= lme.Coefficients.Estimate(1) + iris_data.SepalLength(indx_virginica).*lme.Coefficients.Estimate(2) + rand_eff(3);

% plot new regression model
subplot(1,2,2)
scatter(iris_data.SepalLength, iris_data.SepalWidth, 20, categorical(iris_data.Species), 'filled')
hold on
plot(aFitted1, bFitted1, '--', 'LineWidth', 3, 'Color', [0.4 0.4 0.4])
plot(aFitted2, bFitted2, '--', 'LineWidth', 3, 'Color', [0.4 0.4 0.4])
plot(aFitted3, bFitted3, '--', 'LineWidth', 3, 'Color', [0.4 0.4 0.4])
xlabel('Sepal Length', 'FontSize', 20);
ylabel('Sepal Width', 'FontSize', 20);
hold off;

% What do you notice about this model. How are the interceopts different
% across species? How are the slopes different across species? How is this
% model better? how is it worse?


% Lets go even further and add random slopes
formula         = 'SepalWidth ~ SepalLength+ (SepalLength| Species) ';
tbl             = table(iris_data.SepalLength,iris_data.SepalWidth,iris_data.Species,  'VariableNames',{'SepalLength','SepalWidth','Species'});
lme             = fitlme(tbl,formula);

rand_eff=lme.randomEffects;
aFitted1= iris_data.SepalLength(indx_setosa);
bFitted1= lme.Coefficients.Estimate(1) + iris_data.SepalLength(indx_setosa).*lme.Coefficients.Estimate(2) + rand_eff(1) + rand_eff(2)*iris_data.SepalLength(indx_setosa);

aFitted2= iris_data.SepalLength(indx_versicolor);
bFitted2= lme.Coefficients.Estimate(1) + iris_data.SepalLength(indx_versicolor).*lme.Coefficients.Estimate(2) + rand_eff(3) + rand_eff(4)*iris_data.SepalLength(indx_versicolor);

aFitted3= iris_data.SepalLength(indx_virginica);
bFitted3= lme.Coefficients.Estimate(1) + iris_data.SepalLength(indx_virginica).*lme.Coefficients.Estimate(2) + rand_eff(5) + rand_eff(6)*iris_data.SepalLength(indx_virginica);

% plot effects of random slopes
subplot(1,2,2)
scatter(iris_data.SepalLength, iris_data.SepalWidth, 20, categorical(iris_data.Species), 'filled')
hold on
plot(aFitted1, bFitted1, '--', 'LineWidth', 3, 'Color', [0.4 0.4 0.4])
plot(aFitted2, bFitted2, '--', 'LineWidth', 3, 'Color', [0.4 0.4 0.4])
plot(aFitted3, bFitted3, '--', 'LineWidth', 3, 'Color', [0.4 0.4 0.4])
xlabel('Sepal Length', 'FontSize', 20);
ylabel('Sepal Width', 'FontSize', 20);
hold off;

% what is different now from the last model? How is this one better and
% worse? When would you use this model in comparison to the random
% intercepts? What about the non-hierarchical model?


% some examples of more complex heirarchical models 
date=readtable('./justafolderwithdata/SpeedDatingData.csv');

formula         = 'attr_o ~samerace+ age_o+ age+ income+ exphappy + expnum + (1|Participantid)';
tbl             = table(date.ParticipantId,date.samerace,date.age_o, date.age, date.income,date.attr_o, date.exphappy, date.expnum, 'VariableNames',{'Participantid','samerace','age_o', 'age', 'income', 'attr_o', 'exphappy', 'expnum'});
lme             = fitlme(tbl,formula);


formula         = 'attr_o ~samerace+ age_o+ age+ income+ exphappy + expnum + (1|Wave)';
tbl             = table(date.wave,date.samerace,date.age_o, date.age, date.income,date.attr_o, date.exphappy, date.expnum, 'VariableNames',{'Wave','samerace','age_o', 'age', 'income', 'attr_o', 'exphappy', 'expnum'});
lme             = fitlme(tbl,formula);

formula         = 'match ~samerace+ age_o+ age+ income+ exphappy + expnum + (1|Wave)';
tbl             = table(date.wave,date.samerace,date.age_o, date.age, date.income,date.match, date.exphappy, date.expnum, 'VariableNames',{'Wave','samerace','age_o', 'age', 'income', 'match', 'exphappy', 'expnum'});
glme             = fitglme(tbl,formula,'Distribution','binomial');


% exercise 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% using everything you've learnt build a complex model with the speed
% dating dataset with the goal of explaining the most variance with the
% LEAST number of predictors! 

%%  Model Fit & Selection Reprised: AIC & BIC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% let us fit a bunch of comepting models 
date=readtable('./justafolderwithdata/SpeedDatingData.csv');

formula         = 'attr_o ~samerace+ age_o+ age+ (1|Participantid)';
tbl             = table(date.ParticipantId,date.samerace,date.age_o, date.age, date.income,date.attr_o, date.exphappy, date.expnum, 'VariableNames',{'Participantid','samerace','age_o', 'age', 'income', 'attr_o', 'exphappy', 'expnum'});
lme1             = fitlme(tbl,formula);

formula         = 'attr_o ~samerace+ age_o+ age+ income+  (1|Participantid)';
tbl             = table(date.ParticipantId,date.samerace,date.age_o, date.age, date.income,date.attr_o, date.exphappy, date.expnum, 'VariableNames',{'Participantid','samerace','age_o', 'age', 'income', 'attr_o', 'exphappy', 'expnum'});
lme2             = fitlme(tbl,formula);

formula         = 'attr_o ~samerace+ age_o+ age+ income+ exphappy + expnum + (1|Participantid)';
tbl             = table(date.ParticipantId,date.samerace,date.age_o, date.age, date.income,date.attr_o, date.exphappy, date.expnum, 'VariableNames',{'Participantid','samerace','age_o', 'age', 'income', 'attr_o', 'exphappy', 'expnum'});
lme3             = fitlme(tbl,formula);
% note that normally we test models by adding A SINGLE predictor at a time,
% this is done because we can directly compare the effect of adding it to
% the fit rather than trying to change a bunch of things at once 


% now let us compute their AIC and BIC
% See lecture for details 
logL = [lme1.LogLikelihood; lme2.LogLikelihood; lme3.LogLikelihood];
numParam = [lme1.NumPredictors; lme2.NumPredictors; lme3.NumPredictors];
numObs = length(date.ParticipantId);
aic = aicbic(logL,numParam) % requires Econometrics Toolbox
aic=(-2*logL + 2*numParam)
bic=(-2*logL + numParam*log(numObs))


% exercise 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% of the models we tested which is best and why? Do AIC AND BIC agree? If
% not why would this be? 

% take your really complex model and test it on the same data. What
% happens? (HINT use the predict function in MATLAB). What happens when you
% run your complex regression on a subset of data and save the rest for
% prediction?

%%  Support Vector machines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load and prep data for SVM classifier 
% data for classifers are typically saved as a testing and training set,
% this is done by dividing the original dataset into smaller sets. There
% are many schemes and convensions for this. Do some research on what is
% typical in your field. 

tbl=table2array(tbl);
[train, idx]=datasample(tbl, floor(2*length(tbl)./3), 'Replace', false);
test=tbl(setdiff(1:length(tbl),idx),:);
classes_train=date.match(idx);
classes_test=date.match(setdiff(1:length(tbl),idx));

%Train the SVM Classifier with defaults 
cl = fitcsvm(train,classes_train)
sum(classes_test==predict(cl, test))./length(classes_test)


% since we randomly split the data into a training and testing set, this
% can give us different effects if we run it again. Try running the above
% code again and see what accuracy you get.

% Typically you would run multiple iterations of the training and testing
% such that any biases from splitting your data can be accounted for
numiter=50;
% cross validation
for i=1:numiter
disp(i)
[train, idx]=datasample(tbl, floor(2*length(tbl)./3), 'Replace', false);
test=tbl(setdiff(1:length(tbl),idx),:);
classes_train=date.match(idx);
classes_test=date.match(setdiff(1:length(tbl),idx));

%Train the SVM Classifier
cl = fitcsvm(train,classes_train);
acc(i)=sum(classes_test==predict(cl, test))./length(classes_test);
end


% SVMs trained on a specific dataset (i.e, at one time point) can also be
% used to predict other time points. This is called temporal generalization
