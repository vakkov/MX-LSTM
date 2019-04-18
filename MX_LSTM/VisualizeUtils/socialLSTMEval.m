allSets={'zara01','zara02','stu03','tc'};
datasets=allSets{1};
addpath('dataFiles')
addpath('utils')
if strcmp(datasets,'zara01')
    addpath('/export/work/i.hasan/LSTM_experiments/Social_LSTM/social_lstm/VisualizeUtils/frames_zara01/')
    load('z1_new_bencmark.mat')
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
   % load('z2_nbr_size_dist_changed_ped_wise_social_lstm.mat')
    %load('z2_ped_wise_social_lstm.mat')
    load('z2_ped_wise_social_lstm.mat')
    load('D_zara01.mat')
    load('z2_norm_zer_zero_firstLoc')
    arrayT=csvread('info_z2.csv');
    origFrames=unique(arrayT(:,1));
    normFrame=1:1:length(origFrames);
    fileSaveName='z2_social_lstm.mat';
    normParams.sig(1)=sX;normParams.sig(2)=sY;normParams.mu(1)=deltaX;normParams.mu(2)=deltaY;
elseif strcmp(datasets,'stu03')
    addpath('frames_stu03/')
    load('ucy_ped_wise_social_lstm.mat')
    load('D_ucy.mat')
    load('ucy_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_ucy.csv');
    origFrames=unique(arrayT(:,1));
    normFrame=1:1:length(origFrames);
    fileSaveName='ucy_social_lstm.mat';
    normParams.sig(1)=sX;normParams.sig(2)=sY;normParams.mu(1)=deltaX;normParams.mu(2)=deltaY;
end
genVisualization=1;

counter=1;
social_lstm_mat=[];
FADErr=[];
for ii=1:size(data,1)
    thisGt=[];
    xMatGT=[];
    yMatGT=[];
    thisGt=data{ii,1};
    xMatGT=thisGt(:,:,2);
    yMatGT=thisGt(:,:,3);
    thisDt=[];
    xMatDT=[];
    yMatDT=[];
    thisDt=data{ii,2};
    xMatDT=thisDt(:,:,2);
    yMatDT=thisDt(:,:,3);
    infoPed=data{ii,4};
    pedId=infoPed(:,:,1);
    frameInfo=infoPed(:,:,2);
    thisPedID=data{ii,5}(4,:)'; % extract only pedetsrian id
    myPed=unique(thisPedID);
    for jj=1:length(myPed)
        thisPed=myPed(jj);
        [r,c]=find(pedId==thisPed);
        linInd=sub2ind(size(pedId),r,c);
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
            gtPts=[];
            exGt=[];
            exPred=[];
            thisXDt=[];
            thisYDt=[];
            gtPts(:,1)=xMatGT(linInd);
            gtPts(:,2)=yMatGT(linInd);
            thisXDt=xMatDT(linInd);
            thisYDt=yMatDT(linInd);
            exGt(:,1)=(gtPts(:,1)*sX)+deltaX;
            exGt(:,2)=(gtPts(:,2)*sY)+deltaY;
            exPred(:,1)=(thisXDt(:,1)*sX)+deltaX;
            exPred(:,2)=(thisYDt(:,1)*sY)+deltaY;
            frameId=frameInfo(linInd);
            frameId=refFrame2origFrame(origFrames,normFrame,frameId);
            
            thisFrame=frameId(1);
            tid=find(allPedAnno(:,1)==thisFrame);
            tuple=allPedAnno(1:tid,[3 4]);
            ptsTill=cumsum(tuple,1);
            fristLoc=ptsTill(end,:);
            exGt(1,:)=fristLoc;
            exPred(1,:)=fristLoc;
            gp=[];
            ph=[];
            gp=cumsum(exGt);
            ph=cumsum(exPred);
            err = mean(sqrt(sum((gp(9:end,:) - ph(9:end,:)).^2,2)));
            finalErr = mean(sqrt(sum((gp(end,:) - ph(end,:)).^2,2)));
            meanErr(counter,1)=err;
            FADErr(counter,1)=finalErr;
            finalErr=[];
            social_lstm_mat(counter,1)=thisPed;
            social_lstm_mat(counter,2)=err;
            counter=counter+1;
            if genVisualization
                gtIm=w2i(gp,D.H);
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
end
mean(meanErr)
mean(FADErr)
mean(social_lstm_mat(:,2));
save(fileSaveName,'social_lstm_mat');