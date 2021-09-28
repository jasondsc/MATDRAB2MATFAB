%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               Class 7 
%              Curve and Model Fitting, & Model Selection
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

%%  Polynomial and curve fitting in MATLAB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lets fit a bunch of polynomials to our data 
load('./justafolderwithdata/Iris_2021_data.mat');
figure
for i=1:8
[f, gof, output]=fit(iris_data.InstagramLikes, iris_data.Availability, strcat('poly', int2str(i)))
fits{i}=f;
fits_gof{i}=gof;
fits_out{i}=output;
subplot(2,4,i)
scatter(iris_data.InstagramLikes, iris_data.Availability,20)
hold on
plot(f)
end

[f, gof, output]=fit(iris_data.InstagramLikes, iris_data.Availability, 'poly')

[P,S]=polyfit(iris_data.InstagramLikes, iris_data.Availability,200)

[P,S]=polyfit(iris_data.InstagramLikes, iris_data.Availability,2)

% you can visually pick which model you think best fits the data by looking
% at the figure. Which one would you pick?

% of the top two models based on your visual inspectiopn look at the values
% of the polynomial coefficents

%%  Error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% let us take the above example and compute some different error terms

% error (simply put observed-predicted data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for the first polynimoal we get 
fitted_data1= fits{1}.p1 *iris_data.InstagramLikes + fits{1}.p2;
error1 = iris_data.Availability-fitted_data1;

mean(error1)

% inspect the error variable, now check fits{1}.residuals, what do you
% notice

fitted_data2= fits{4}.p1 *iris_data.InstagramLikes.^4 + fits{4}.p2 *iris_data.InstagramLikes.^3 + fits{4}.p3 *iris_data.InstagramLikes.^2 + fits{4}.p4 *iris_data.InstagramLikes + fits{4}.p5;
error2 = iris_data.Availability-fitted_data2;

mean(error2)

% square error 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this squares the error term to take into account the time when your model
% over or underestimated the observed data point

sqerror1 = (iris_data.Availability-fitted_data1).^2;
sqerror2 = (iris_data.Availability-fitted_data2).^2;

% take the mean of the squared errors 
mean(sqerror1)
mean(sqerror2)

% take the sum of squered errors (SSE)
sum(sqerror1)
sum(sqerror2)


% root Mean Square error 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simply put this is the square root of the mean of squared errors (see
% above) 

% take the mean of the squared errors 
sqrt(mean(sqerror1))
sqrt(mean(sqerror2))


% Compute R2 coefficient of determination 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R2 represents the porportion of the total variance in your data that is
% explained by your model/ predictors 

% compute total sum of squares 
SStot= sum((iris_data.Availability-mean(iris_data.Availability)).^2);

% compute sum of squared residuals 
SSres1 =sum(sqerror1);
SSres2 =sum(sqerror2);

% compute R2
r21 = 1- (SSres1/SStot)
r22 = 1- (SSres2/SStot)


% adjusted R2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tries to control for the number of parameters in your model see the
% lecture for details on why this is importnat 

radj1= 1 - (1-r21)*((length(iris_data.Availability)-1)/(length(iris_data.Availability)-fits_out{1}.numparam-1))

radj2= 1 - (1-r22)*((length(iris_data.Availability)-1)/(length(iris_data.Availability)-fits_out{4}.numparam-1))


%%  Log Likelihoods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


rng default;  % For reproducibility
a = [1,2];    % let us make a normal dist with a mean of 1 and std of 2
% these will act as our two parameters we will try and estimate using logL

% generate some random samples/ data with these params
X = normrnd(a(1),a(2),1e3,1);

mesh = 50; % define the size of the meshgrid we are seaching through 
delta = 0.5; % define the upper and lower limits of the mesh
% this will be the grid or parameter space we explore (i.e., 1+/- 0.5 )
a1 = linspace(a(1)-delta,a(1)+delta,mesh);
a2 = linspace(a(2)-delta,a(2)+delta,mesh);
logL = zeros(mesh); % Preallocate memory
% iterate through mesh grid / parameter space (x and y values of matrix)
for i = 1:mesh
    for j = 1:mesh
        logL(i,j) = normlike([a1(i),a2(j)],X); % compute the log like for said parameter combo
    end
end
 
[A1,A2] = meshgrid(a1,a2); % make mesh grid for plotting 
% plot parameter space (i.e, 3D spcae, param 1, param2, and LogL)
figure
surfc(A1,A2,logL)
hold on;
% Try and fit a normal dist for your random data X
[est_mu, est_sigma]=normfit(X)

% how about another algorithm for searching parameter spaces?
% Fmin search is commonly used to solve equations

% first you need to define the function (or surface) we want to find the
% minimum of (i.e., the log likelihood function) 
LL = @(u)normlike([u(1),u(2)],X); 
MLES = fminsearch(LL,[1.1,2.2]) % search the space (i.e., function) using the starting values

% try switching the starting values X0 [1 ,2] what happens to the end
% result?

% find location of closest value of estimated param
[temp location_x]=min(abs(A1(1,:)-est_mu));
[temp location_y]=min(abs(A2(:,1)-est_sigma));
% plot the estimated value of the param from the fitting
plot3(A1(1,location_x), A2(location_y,1), logL(location_x,location_y), 'ro','MarkerSize',10,'MarkerFaceColor','r');

% find location of closest value of estimated param
[temp location_x]=min(abs(A1(1,:)-MLES(1)));
[temp location_y]=min(abs(A2(:,1)-MLES(2)));
plot3(A1(1,location_x), A2(location_y,1), logL(location_x,location_y), 'mo','MarkerSize',10,'MarkerFaceColor','r');

% What do you notice from the above graph? Did the algorithm do a good job
% at fitting the data? What happens if you expand the parameter space? What
% happens if you generate more data and try and fit the param again? Is it
% better? Is there a balance between the param space and the quantity of
% data 


% exercise 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the online dating dataset and fit a model (polynomial) to the
% relationship between profile visits and languages spoken. What is the best fitting
% model? What is the order of the relationship? Do all error metrics show
% similar findings? 

dating=readtable('./justafolderwithdata/modified_lovoo_v3_users_instances.csv');

dating=dating(:,[2:8 10:end-1]);

figure
scatter(dating.counts_profileVisits, dating.age)

[f, gof, output]=fit(dating.counts_profileVisits, dating.age, strcat('poly', int2str(9)))


% Try removing outliers from the data and refit the models, what is
% different? 


rm_dating=rmoutliers([dating.counts_profileVisits, dating.age])
figure
scatter(rm_dating(:,1), rm_dating(:,2))

%%  Linear Regression Models 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Let us try regressing the effects of species on sepal length 

load fisheriris

% table all the variables we wish to explore into one large data frame
data=table(meas(:,1), meas(:,2),...
        meas(:,3), meas(:,4),...
      species);
data.Properties.VariableNames = {'Speal_Length' 'Sepal_Width' 'Petal_Length' 'Petal_Width', 'Species'} % rename columns

% let us effect code the data (i.e., turn the species into numbers)
data.Species_coded(find(strcmp(data.Species, 'virginica'))) = .5;
data.Species_coded(find(strcmp(data.Species, 'setosa'))) = 0;
data.Species_coded(find(strcmp(data.Species, 'versicolor'))) = -.5;

% now that we have rearranged our data and put it in a format that we can
% work with (i.e., long format) let us run a regression model 
X = table(ones(size(data.Species)), data.Species_coded);
X.Properties.VariableNames = {'Intercept', 'Species'};


[b,bint,r,rint,stats] = regress(data.Sepal_Width,table2array(X))    % Removes NaN data
% b will return the beta coefficents
% bint will retrun the 95% CI of the betas
% r returns the residulas
% rint diagnostic intervals to check for outliers 
% stats returns the R2, the F, the p-value, and the estimated error
% variance

% let us explore b
disp(b) 

% now let us run an ANOVA and compare the effects
[p,tbl, stats]= anova1(data.Sepal_Width, data.Species)
% exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% what do you notice? How are they the same? How are they different? Is a
% regression an ANOVA? Is an ANOVA a regression? What can one do that the
% other can't? 

