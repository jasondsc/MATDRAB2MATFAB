
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                 Class 1 & 2
%                               Basics of MATLAB 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Basic MATLAB commands
pwd                                                     % find your current directory 
cd('./justafolderwithdata/')             % Change your current directory 
ls                                                      % find all the files in your directory 
cd('../')                                           % moves you to the directory above 
which mean

listoffiles=dir("./justafolderwithdata");
what('./justafolderwithdata/')
exist('./justafolderwithdata/raw_spike_data.mat')
load('./justafolderwithdata/raw_spike_data.mat')

close all                                           % clears your workspace environemnt 
clear all                                            % closes all figures
clc                                                     %clears command window



%%  Variables 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

var=1;
var2=2;
var3=1:300;

max([var, var2])
[maxA, location]=max(var3);
plot(var3, '-or')

help plot
help pca

bool=true;
bool=false;
ints=1:12;
bools= boolean([ 1 0 1 0 1 1 1 1 0 1 0 1]);
ints(bools)

bool1=[1 0 1 0 1]
bool2=[1 0 0 1 1]

bool1== bool2
bool1 & bool2
bool1 | bool2
bool1(1) && bool2(1)
bool1(2) || bool2(2)

% there are different ways to store and express numbers (i.e., numerics) 
num=single(2)
num2=double(2)
num3=uint8(2)

cast(num, 'double')

%%  Strings 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% differences between strings and cahrs
secret_message = ['Homer';  'Olaf' ; 'Nathan'; 'Ethan'; 'Yusef']
secret_message = ["Homer";  "Olaf" ; "Nathan"; "Ethan"; "Yusef"]

secret_message=['hel'; 'ooo']

whenDoUgetPresents="Christmas"

word0 = "My" ; word0(1)
word1= 'My' ; word1(2)
string2= "All" ; string2{1}(2)
string3= "hello my name is"; string3{1}

word1 ='We'
word2='belong'
word3='together'

str1 = "Come back"
str2 =" baby"
str4=str1+ str2

strcmp("Always be my baby", 'Always be my baby')

isItHardtoBelieve=true;
ComeBackBabyPls=" please";

song=strcat(" When you left I lost a part of me", int2str(isItHardtoBelieve), str4, ComeBackBabyPls, " because ", strjoin({word1, word2, word3}))

replace(song, "1", ", its still so hard to believe ")

splitstr=split(song)

new_secret=strjoin({secret_message{5}(5), secret_message{1}(2), secret_message{1}(5)},'')    

secret_message=strcat(string2, song{1}(16),  'Want', new_secret,whenDoUgetPresents ,int2str(15),splitstr{3})


%% Arrays 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Making an array
ar=[]; % empty array
ar=[1,2,3,4,5,6,7]; % array of ints
ar=[1,2,3;4,5,6;7,8,9]; % matrix array of ints
z=zeros(10,10);
z=z+10;
z=z-9;
z=z*3;

z.*z % IS DIFFERENT THAN ' * '

z*z % THIS IS THE MATRIX PRODUCT OR DOT PRODUCT

y=ones(1,10)+13;
z*y'

% MATRIX AND ARRAYS
a=randi([0,100],10,10); % make random matrix of intergers 
a(1,10) % index first row, 10th col
a(:,5) % index 5th col
a(1,1:5) % 1st element of the 1st 5 cols

% Mili-dimensional arryas
multi=randi([-100,100],30,121,64); % make random matrix of intergers 

multi=randi([-100,100],30,121,64,10); % make random matrix of intergers 

% Reshaping a Matrix
a=randi([0,100],10,10); % make random matrix of intergers 
reshape(a, [10,2,5])
reshape(a, [100,1])
reshape(a, [2,50])


%%  File I/O Basics
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
load("./justafolderwithdata/raw_spike_data.mat")
clear all
close all

cd("./justafolderwithdata/")
load('raw_spike_data.mat')
save("temp_data.mat") % saves workspace to file 

image=imread('./MemeFolder/ladygaga5.jpg');
imagesc(image)

table=readtable('./justafolderwithdata/RPDR_contestant_score.csv');

% if you are saving large ammounts of data >2.0GB, use -v7.3 specifier 
save('./workspace.mat', '-v7.3')

writetable(table, 'test_output.txt','Delimiter',',')

csvwrite('image.txt', mean(image,3))

%read csv, tsv, or txt (note that the values must all be numeric) 
%   csvread()
%   tdfread()

% Note that files can be imported using the GUI of MATLAB as well...
% % drag and drop .mat files into the command window to load them
% % right click on a variable to save them as .mat 
% % use the import tool in the HOME tab to read data of many file formats


%%  Exercises class 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1) read the RPDR_contestant_data and compute the mean age of all contestants 

table=readtable('./justafolderwithdata/RPDR_contestant_data.csv');

mean(table2array(table(:,4)))

% 2) How many special episodes were there on RPDR? How many Finale
% episodes? 

epi=readtable('./justafolderwithdata/RPDR_episode_data.csv');

sum(table2array(epi(:,5)))

% 3) What was the largest number of epsidoes for a season?
% Note you need to read about cells and for loops below !

c={'S01','S02','S03','S04'};
for i=1:4
     max(epi.episode(strcmp(epi.season, c{1,i})))
     
end

% hint use strcmp()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%                        `         Class 2
%                       Objects, Loops, Conditionals 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Set Operators
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


set1=randi([0,100],100,1);
set2=randi([-50,50],100,1);

ismember(1,set1)
ismember([1,2,3,4,5],set1)

intersect(set1,set2)
intersect(set2,set1)
setdiff(set1,set2)
setdiff(set2,set1)

union(intersect(set1,set2),setdiff(set1,set2))

unique(set1)

set1 = {"homer";  "Olaf" ; "Nathan"; "David"; "Yusef"}
set2 = {"Homer";  "Nathan"; "Ethan"; "Yusef"}
set3= ["Homer";  "Nathan"; "Ethan"; "Yusef"]

ismember("homer",set1)
ismember("homer", set3)

% Note that for strings it must be a string ARRAY, character VERCTOR or a
% CELL ARRAY of chars
set1 = ["AIW4CIU";  "Joy to the world" ; "Open arms"; "Honey"; "We belong together"; "Emotions"; "Vision of Love"; "My All"; "forever" ]

set2 = ["Honey";  "Emotions"; "Always be my baby"; "we belong together"; "Fantasy"; "Obsessed"]

ismember("we belong together",set1)
intersect(set1,set2)
setdiff(set1,set2) 
setdiff(set2,set1) 

unique([set1;set2])

%% Data Structures  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% tables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
listoffiles=dir("./");

drag_race=readtable('./justafolderwithdata/RPDR_contestant_data.csv')
[uniquenames, ids, ~]=unique(drag_race.contestant)
drag_race=drag_race(ids,:)
drag_race = sortrows(drag_race,'contestant','ascend');
score=readtable('./justafolderwithdata/RPDR_contestant_score.csv')
sum(strcmp(score.contestant, drag_race.contestant))
full_drag=[drag_race, score(:,3:7)]
StructArray = table2struct(full_drag);

table2array(score(:,3:7))

% structs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% used in object oriented programming, useful when you want an object to
% have several features

% Lets make a student struct
student(1).name="Jason";
student(1).age=25;
student(1).GPA= 3.97;
student(1).FavMariahSong= "We belong together";
student(1).buyJoanneoniTunes= true;
student(1).single=true;
student(1).thesisdata=rand(100,100);

student(2).name="Joe";
student(2).age=25;
student(2).GPA= 3.97;
student(2).FavMariahSong= "We belong together";
student(2).buyJoanneoniTunes= true;
student(2).single=true;
student(2).thesisdata=rand(100,100);

struct2table(student)

student(3).name=44;
student(3).age=25;
student(3).GPA= 3.97;
student(3).FavMariahSong= "We belong together";
student(3).buyJoanneoniTunes= true;
student(3).single=true;
student(3).thesisdata=rand(100,100);

struct2table(student)

study.drag_race=drag_race;
study.sampling_rate=1000;
study.eeg=rand(100,1000);

save('filename.mat', 'study')

% cells
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% are like containers that hold different tupe of info and of different
% sizes!!!!
c={randi([0,100],10,10), 'hello';randi([0,100],5,5), 2;randi([0,100],2,2),2000; "MARIAH CAREY RULES", true};

cellplot(c)

c={student;study;drag_race; 'MARIAH CAREY RULES'};
cellplot(c)

c={randi([0,100],10,1),randi([0,100],5,1),randi([0,100],2,1)}

cellfun(@mean, c)


c={randi([0,100],10,10),randi([0,100],5,5),randi([0,100],2,2)}

cellfun(@mean, c)

cellfun(@mean, c, 'UniformOutput', false)

C = {'Monday','Tuesday','Wednesday','Thursday','Friday'}
cellfun(@(x) x(1:3),C,'UniformOutput',false)

c={randi([0,100],10,10),randi([0,100],5,5),randi([0,100],2,2);...
randi([0,100],30,10),randi([0,100],5,25),randi([0,100],22,2)}

cellfun(@mean, c,'UniformOutput',false)


%% Loops and Conditionals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Conditionals if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a=randi(100,1);

if a< 30
    disp("she's small")
elseif a <70
    disp("she's okay")
else
    disp("SHES HUGE")
end

if a< 30
        disp("she's small")
else
    disp("SHES HUGE")
end

if a< 30
    disp("she's small")
elseif a <70
    disp("she's okay")
end

% Conditionals Switch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dayNum, dayString] = weekday(date, 'long', 'en_US');
dayString='Oct';
tic
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
case 'Saturday'
    disp("weekend")
case 'Sunday'
    disp("weekend")
otherwise
    disp('YAS BITCH WERK')
end
toc
% For Loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
arr=rand(1,100);

for i= 1:2:100
        if mod(i,9)==0
            disp(i)
            disp(" is a multiple of 9!!")
        end
end

for i= 1:2:100
    
    for j= 1:100
        if mod(i,9)==0
            disp(i)
            disp(" is a multiple of 9!!")
        end
    end
end


% While loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=1;
nfactorial=1;

while nfactorial <500
    n=n+1;
    nfactorial= nfactorial*n;
end




%% Exercises class 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1) move to the data_from_sub directory, loop through files, select the ones you
% want, open the file, extract the data, and save it! Plot a histogram of
% data

cd './justafolderwithdata/data_from_sub'

list=dir('sub*');

for i = 1:length(list)
    
    load(list(i).name)

    newdata(i,:,:)=data;
    
    
end


hist(reshape((newdata), [20*20*200,1]))


% 2) using the full_drag table, compute how many different cities are
% represented on drag race

length(unique(drag_race.hometown))


% 3) using the full_drag table, compute a histogram of the age of
% participants

hist(drag_race.age)



