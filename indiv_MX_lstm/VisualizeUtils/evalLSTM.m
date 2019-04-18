clearvars
addpath('dataFiles')
addpath('utils')
addpath('output')
allSets={'zara01','zara02','stu03'};
datasets=allSets{3};
if strcmp(datasets,'zara01')
addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara01\')
%% name of the output file below by running test_pedestrian_wise
load('z1_in_correct_vislet.mat')
%% -------------------------------------------------------
load('D_zara01.mat')
load('z1_norm_zer_zero_firstLoc.mat')
arrayT=csvread('info_z1.csv');
fileSaveName='z1_vanila_lstm.mat';
elseif strcmp(datasets,'zara02')
addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara02\')
%% name of the output file below by running test_pedestrian_wise
load('z2_in_correct_vislet.mat')
%% -------------------------------------------------------
load('D_zara01.mat')
load('z2_norm_zer_zero_firstLoc.mat')
arrayT=csvread('info_z2.csv');
fileSaveName='z2_vanila_lstm.mat';
elseif strcmp(datasets,'stu03')
addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_stu03\')
%% name of the output file below by running test_pedestrian_wise
load('ucy_in_correct_vislet.mat')
%% -------------------------------------------------------
load('D_ucy.mat')
load('ucy_norm_zer_zero_firstLoc.mat')
arrayT=csvread('info_ucy.csv');
fileSaveName='ucy_vanila_lstm.mat';
end
genVisualization=0;
allPed=unique(arrayT(:,2));
meanErr=[];
w=720;
h=576;
meanErr=[];
FADErr=[];
im=zeros(h,w);
van_mat=[];
for ii=1:length(data)
    thisGt=data{ii,1};
    thisDt=data{ii,2};
    thisPedID=data{ii,4}(:,2); % extract only pedetsrian id
    myPed=unique(thisPedID);
    idx=find(arrayT(:,2)==myPed);
    allPedAnno=arrayT(idx,:);
    dl=diff(allPedAnno(:,3:4));
    dl=[allPedAnno(1,[3 4]);dl];
    allPedAnno(:,[3 4])=dl;
    frameId=data{ii,4}(:,1); % extract only frameId
    gidx=find(firstLocs(:,1)==myPed);
    psStart=firstLocs(gidx,2:3);
    phat=thisDt(:,1:2);
    gtPts=thisGt(:,1:2);
    exGt(:,1)=(gtPts(:,1)*sX)+deltaX;
    exGt(:,2)=(gtPts(:,2)*sY)+deltaY;
    exPred(:,1)=(thisDt(:,1)*sX)+deltaX;
    exPred(:,2)=(thisDt(:,2)*sY)+deltaY;
    thisFrame=frameId(1);
    tid=find(allPedAnno(:,1)==thisFrame);
    tuple=allPedAnno(1:tid,[3 4]);
    ptsTill=cumsum(tuple,1);
    fristLoc=ptsTill(end,:);
    exGt(1,:)=fristLoc;
    exPred(1,:)=fristLoc;
    gp=cumsum(exGt);
    ph=cumsum(exPred);
    gtIm=w2i(gp,D.H);
    predIm=w2i(ph,D.H);
    if genVisualization
        im=imread(sprintf('%06d.png',frameId(1)));
        figure
        imshow(im), hold on
        plot(gtIm(:,1),gtIm(:,2),'g*')
        plot(predIm(9:end,1),predIm(9:end,2),'b*')
        pause(0.8)
    end
    err = mean(sqrt(sum((gp(9:end,:) - ph(9:end,:)).^2,2)));
    finalErr = mean(sqrt(sum((gp(end,:) - ph(end,:)).^2,2)));
    meanErr(ii,1)=err;
    FADErr(ii,1)=finalErr;
    idsPed(ii,1)=unique(thisPedID);
    exGt=[];
    exPred=[];
    
end

mean(meanErr)
mean(FADErr)



