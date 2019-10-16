allSets={'zara01','zara02','stu03','tc'};
datasetToPick=3;
datasets=allSets{datasetToPick};

addpath('dataFiles')
addpath('utils')


allOutputFileName={'z1_benchmark_mx_lstm.mat', 'z2_benchmark_mx_lstm.mat','ucy_benchmark_mx_lstm.mat'}; % saved outfilename

pythonOutputFileName = allOutputFileName{datasetToPick};
if strcmp(datasets,'zara01')
    addpath('/export/work/i.hasan/LSTM_experiments/Social_LSTM/social_lstm/VisualizeUtils/frames_zara01/')
    load(pythonOutputFileName)
    load('D_zara01.mat')
    load('z1_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_z1.csv');
    origFrames=unique(arrayT(:,1));
    normFrame=1:1:length(origFrames);
    frameDiff=8;
    fileSaveName='z1_social_lstm.mat';
    normParams.sig(1)=sX;normParams.sig(2)=sY;normParams.mu(1)=deltaX;normParams.mu(2)=deltaY;
elseif strcmp(datasets,'zara02')
    addpath('frames_zara02/')
    load(pythonOutputFileName)
    load('D_zara01.mat')
    load('z2_norm_zer_zero_firstLoc')
    arrayT=csvread('info_z2.csv');
    origFrames=unique(arrayT(:,1));
    normFrame=1:1:length(origFrames);
    fileSaveName='z2_social_lstm.mat';
    normParams.sig(1)=sX;normParams.sig(2)=sY;normParams.mu(1)=deltaX;normParams.mu(2)=deltaY;
elseif strcmp(datasets,'stu03')
    addpath('frames_stu03/')
    load(pythonOutputFileName)
    load('D_ucy.mat')
    load('ucy_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_ucy.csv');
    origFrames=unique(arrayT(:,1));
    normFrame=1:1:length(origFrames);
    fileSaveName='ucy_social_lstm.mat';
    normParams.sig(1)=sX;normParams.sig(2)=sY;normParams.mu(1)=deltaX;normParams.mu(2)=deltaY;
end
genVisualization=0;
counter=1;

FADErr=[];
for ii=1:size(data,1)
    [gtPts,thisDt,frameId,pedId,thisPed,frameInfo,linInd]=extractNomralizedTraj(data,ii);
    idx=find(arrayT(:,2)==thisPed);
    allPedAnno=arrayT(idx,:);
    dl=diff(allPedAnno(:,3:4));
    if(size(dl,1)>1)
        dl=[allPedAnno(1,[3 4]);dl];
    else
        dl=[0 0];
    end
    allPedAnno(:,[3 4])=dl;
    if length(linInd)==20
       [gp,ph]=denomPts(gtPts,thisDt,normParams,allPedAnno,origFrames,normFrame,frameId);
        err = mean(sqrt(sum((gp(9:end,:) - ph(9:end,:)).^2,2)));
        finalErr = mean(sqrt(sum((gp(end,:) - ph(end,:)).^2,2)));
        MADErr(counter,1)=err;
        FADErr(counter,1)=finalErr;
        finalErr=[];
        counter=counter+1;
        if genVisualization
            gtIm=w2i(gp,D.H); % in image plane
            predIm=w2i(ph,D.H);
            im=imread(sprintf('%06d.png',frameId(1)));
            figure
            imshow(im), hold on
            plot(gtIm(:,1),gtIm(:,2),'g*')
            plot(predIm(9:end,1),predIm(9:end,2),'b*')
            pause(0.8)
        end
    end
    
end
mean(MADErr)
mean(FADErr)

