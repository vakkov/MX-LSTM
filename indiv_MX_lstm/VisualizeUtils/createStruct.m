clearvars
addpath('dataFiles')
addpath('utils')
addpath('output')
allSets={'zara01','zara02','stu03'};
datasets=allSets{1};


correct=0;
if correct
    matFileName='_in_correct_vislet_basleine.mat';
else
    matFileName='_in_correct_vislet_east.mat';
end
matFileName
if strcmp(datasets,'zara01')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara01\')
    %% name of the output file below by running test_pedestrian_wise
    load(strcat('z1',matFileName));
    %% -------------------------------------------------------
    load('D_zara01.mat')
    load('z1_norm_zer_zero_firstLoc_fixed_vislet.mat')
    arrayT=csvread('info_z1_fixed_vislet.csv');
    fileSaveName=strcat('z1_traj_struct',matFileName);
elseif strcmp(datasets,'zara02')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara02\')
    %% name of the output file below by running test_pedestrian_wise
    load(strcat('z2',matFileName));
    %% -------------------------------------------------------
    load('D_zara01.mat')
    load('z2_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_z2.csv');
    fileSaveName=strcat('z2_traj_struct',matFileName);
elseif strcmp(datasets,'stu03')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_stu03\')
    %% name of the output file below by running test_pedestrian_wise
    load(strcat('ucy',matFileName));
    %% -------------------------------------------------------
    load('D_ucy.mat')
    load('ucy_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_ucy.csv');
    fileSaveName=strcat('ucy_traj_struct',matFileName);
end
genVisualization=0;
allPed=unique(arrayT(:,2));
meanErr=[];
w=720;
h=576;
meanErr=[];
FADErr=[];
im=zeros(h,w);
dataStruct=[];
for ii=1:length(data)
    thisGt=data{ii,1};
    thisDt=data{ii,2};
    xMatGTvis=thisGt(:,3);
    yMatGTvis=thisGt(:,4);
    xMatDTvis=thisDt(:,3);
    yMatDTvis=thisDt(:,4);
    thisPedID=data{ii,4}(:,2); % extract only pedetsrian id
    myPed=unique(thisPedID);
    idx=find(arrayT(:,2)==myPed);
    allPedAnno=arrayT(idx,:);
    dl=diff(allPedAnno(:,3:4));
    dl=[allPedAnno(1,[3 4]);dl];
    dlVis=diff(allPedAnno(:,5:6));
    if(size(dl,1)>1)
        dlVis=[allPedAnno(1,[5 6]);dlVis];
    else
        dl=[0 0];
    end
    allPedAnno(:,[3 4])=dl;
    allPedAnno(:,[5 6])=dlVis;
    frameId=data{ii,4}(:,1); % extract only frameId
    gidx=find(firstLocs(:,1)==myPed);
    psStart=firstLocs(gidx,2:3);
    phat=thisDt(:,1:2);
    gtPts=thisGt(:,1:2);
    thisXDtVis=xMatDTvis;
    thisYDtVis=yMatDTvis;
    thisXGtVis=xMatGTvis;
    thisYGtVis=yMatGTvis;
    exGt(:,1)=(gtPts(:,1)*sX)+deltaX;
    exGt(:,2)=(gtPts(:,2)*sY)+deltaY;
    exPred(:,1)=(thisDt(:,1)*sX)+deltaX;
    exPred(:,2)=(thisDt(:,2)*sY)+deltaY;
    exVisGT(:,1)=(thisXGtVis(:,1)*sVX)+deltaVX;
    exVisGT(:,2)=(thisYGtVis(:,1)*sVY)+deltaVY;
    exPredVis(:,1)=(thisXDtVis(:,1)*sVX)+deltaVX;
    exPredVis(:,2)=(thisYDtVis(:,1)*sVY)+deltaVY;
    thisFrame=frameId(1);
    tid=find(allPedAnno(:,1)==thisFrame);
    tuple=allPedAnno(1:tid,[3 4]);
    ptsTill=cumsum(tuple,1);
    fristLoc=ptsTill(end,:);
    tupleVis=allPedAnno(1:tid,[5 6]);
    ptsTillVis=cumsum(tupleVis,1);
    fristLocVis=ptsTillVis(end,:);
    exGt(1,:)=fristLoc;
    exPred(1,:)=fristLoc;
    exVisGT(1,:)=fristLocVis;
    exPredVis(1,:)=fristLocVis;
    gp=cumsum(exGt);
    ph=cumsum(exPred);
    gpVis=cumsum(exVisGT);
    phVis=cumsum(exPredVis);
    gtIm=w2i(gp,D.H);
    predIm=w2i(ph,D.H);
    gtImVis=w2i(gpVis,D.H);
    predImVis=w2i(phVis,D.H);
    if genVisualization
        im=imread(sprintf('%06d.png',frameId(1)));
        figure
        imshow(im), hold on
        for xx=1:size(gtIm,1)-1
            line([gtIm(xx,1) gtIm(xx+1,1)],[gtIm(xx,2) gtIm(xx+1,2)],'color','g','linewidth',2)
            plotVisletOnImagesQuiver(gtIm(xx,1),gtIm(xx,2),gtImVis(xx,1),gtImVis(xx,2),'m')
        end
        predInt=predIm(9:end,:);
        predImVis=predImVis(9:end,:);
        for xx=1:size(predInt,1)-1
            line([predInt(xx,1) predInt(xx+1,1)],[predInt(xx,2) predInt(xx+1,2)],'color','b','linewidth',2)
         %  plotVisletOnImagesQuiver(predInt(xx,1),predInt(xx,2),predImVis(xx,1),predImVis(xx,2),'r')
        end
    end
    err = mean(sqrt(sum((gp(9:end,:) - ph(9:end,:)).^2,2)));
    dataStruct(ii).pedId=repmat(myPed,length(frameId),1);
    dataStruct(ii).frameId=frameId;
    dataStruct(ii).gtIm=gtIm;
    dataStruct(ii).predIm=predIm;
    dataStruct(ii).gtImVis=gtImVis;
    dataStruct(ii).predImVis=predImVis;
    dataStruct(ii).err=err;
    dataStruct(ii).intervalPedId=myPed;
    dataStruct(ii).intervalStartingFrame=frameId(1);
    finalErr = mean(sqrt(sum((gp(end,:) - ph(end,:)).^2,2)));
    meanErr(ii,1)=err;
    FADErr(ii,1)=finalErr;
    idsPed(ii,1)=unique(thisPedID);
    exGt=[];
    exPred=[];
    
end


if correct
    correctData=dataStruct;
    save(fileSaveName,'correctData')
else
    incorrectData=dataStruct;
    save(fileSaveName,'incorrectData')
end




mean(meanErr)
mean(FADErr)



