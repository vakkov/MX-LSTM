allSets={'zara01','zara02','stu03','tc','tc_real_head_pose'};
datasets=allSets{1};
if strcmp(datasets,'zara01')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara01\')
    load('z1_correct_vislet.mat')
    load('D_zara01.mat')
    load('z1_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_z1.csv');
    fileSaveName='z1_vanila_vislet_lstm.mat';
elseif strcmp(datasets,'zara02')
    addpath('frames_zara02/')
    %load('z2_pedestrianwise_zero_zero_vanilla_vislet_lstm.mat')
    load('z2_specific_163_2_ped_pedestrianwise_zero_zero_vanilla_vislet_lstm.mat')
    load('D_zara01.mat')
    load('z2_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_z2.csv');
    fileSaveName='z2_vanila_vislet_lstm.mat';
elseif strcmp(datasets,'stu03')
    addpath('frames_stu03/')
    load('ucy_pedestrianwise_zero_zero_vanilla_vislet_lstm.mat')
    load('D_ucy.mat')
    load('ucy_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_ucy.csv');
    fileSaveName='ucy_vanila_vislet_lstm.mat';
elseif strcmp(datasets,'tc')
    addpath('/export/work/i.hasan/frames/')
    load('tc_tc_only_pedestrianwise_zero_zero_vanilla_vislet_lstm.mat')
    load('D_tc.mat')
    load('tc_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_tc.csv');
    fileSaveName='tc_vanila_vislet_lstm.mat';
elseif strcmp(datasets,'tc_real_head_pose')
    addpath('/export/work/i.hasan/frames/')
    load('tc_real_head_pose_z1_z2_ucy_only_pedestrianwise_zero_zero_vanilla_vislet_lstm.mat')
    load('D_tc.mat')
    load('tc_real_head_pose_norm_zer_zero_firstLoc.mat')
    arrayT=csvread('info_tc_real_head_pose.csv');
end
genVisualization=1;
batchCounter=1;
counter=1;
social_vislet_lstm_mat=[];
thisSpecificPed=1;
thisPedBatch=1;
for ii=1:size(data,1)
    thisGt=[];
    xMatGT=[];
    yMatGT=[];
    thisGt=data{ii,1};
    xMatGT=thisGt(:,1);
    yMatGT=thisGt(:,2);
    xMatGTvis=thisGt(:,3);
    yMatGTvis=thisGt(:,4);
    thisDt=data{ii,2};
    xMatDT=thisDt(:,1);
    yMatDT=thisDt(:,2);
    xMatDTvis=thisDt(:,3);
    yMatDTvis=thisDt(:,4);
    infoPed=data{ii,4};
    pedId=infoPed(:,2);
    frameInfo=infoPed(:,1);
    thisPedID=data{ii,4}(:,2); % extract only pedetsrian id
    myPed=unique(thisPedID);
    if (thisSpecificPed==myPed)
        for jj=1:length(myPed)
            thisPed=myPed(jj);
            [r,c]=find(pedId==thisPed);
            linInd=sub2ind(size(pedId),r,c);
            idx=find(arrayT(:,2)==thisPed);
            allPedAnno=arrayT(idx,:);
            dl=diff(allPedAnno(:,3:4));
            dlVis=diff(allPedAnno(:,5:6));
            if(size(dl,1)>1)
                dl=[allPedAnno(1,[3 4]);dl];
                dlVis=[allPedAnno(1,[5 6]);dlVis];
            else
                dl=[0 0];
            end
            allPedAnno(:,[3 4])=dl;
            allPedAnno(:,[5 6])=dlVis;
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
                thisXDtVis=xMatDTvis(linInd);
                thisYDtVis=yMatDTvis(linInd);
                thisXGtVis=xMatGTvis(linInd);
                thisYGtVis=yMatGTvis(linInd);
                
                exGt(:,1)=(gtPts(:,1)*sX)+deltaX;
                exGt(:,2)=(gtPts(:,2)*sY)+deltaY;
                exPred(:,1)=(thisXDt(:,1)*sX)+deltaX;
                exPred(:,2)=(thisYDt(:,1)*sY)+deltaY;
                exVisGT(:,1)=(thisXGtVis(:,1)*sVX)+deltaVX;
                exVisGT(:,2)=(thisYGtVis(:,1)*sVY)+deltaVY;
                exPredVis(:,1)=(thisXDtVis(:,1)*sVX)+deltaVX;
                exPredVis(:,2)=(thisYDtVis(:,1)*sVY)+deltaVY;
                frameId=frameInfo(linInd);
                %       frameId=refFrame2origFrame(origFrames,normFrame,frameId);
                
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
                gp=[];
                ph=[];
                gp=cumsum(exGt);
                ph=cumsum(exPred);
                gpVis=cumsum(exVisGT);
                phVis=cumsum(exPredVis);
                %
                %                 meanErr(counter,1)=err;
                social_vislet_lstm_mat(counter,1)=thisPed;
                %                social_vislet_lstm_mat(counter,2)=err;
                counter=counter+1;
                if(thisPedBatch==batchCounter)
                    if genVisualization
                        
                        gtIm=w2i(gp,D.H);
                        predIm=w2i(ph,D.H);
                        gtImVis=w2i(gpVis,D.H);
                        predImVis=w2i(phVis,D.H);
                        im=imread(sprintf('%06d.png',frameId(1)));
                        figure
                        imshow(im), hold on
                       for xx=1:size(gtIm,1)-1
                            line([gtIm(xx,1) gtIm(xx+1,1)],[gtIm(xx,2) gtIm(xx+1,2)],'color','g','linewidth',2)
                            plotVisletOnImagesQuiver(gtIm(xx,1),gtIm(xx,2),gtImVis(xx,1),gtImVis(xx,2),'m')
                        end
                        predInt=predIm(9:end,:);
                        predImVis=predImVis(9:end,:);
                      %  plot(predImVis(:,1),predImVis(:,2),'b*')
                        for xx=1:size(predInt,1)-1
                            line([predInt(xx,1) predInt(xx+1,1)],[predInt(xx,2) predInt(xx+1,2)],'color','b','linewidth',2)
                          %  plotVisletOnImagesQuiver(predInt(xx,1),predInt(xx,2),predImVis(xx,1),predImVis(xx,2),'r')
                        end
%                         for xx=1:size(gtIm,1)-1
%                             if xx>=8
%                                 line([gtIm(xx,1),gtIm(xx+1,1)],[gtIm(xx,2),gtIm(xx+1,2)],'color','g','linewidth',2)
%                                 line([predIm(xx,1),predIm(xx+1,1)],[predIm(xx,2),predIm(xx+1,2)],'color','k','linewidth',2)
%                             else
%                                 line([gtIm(xx,1),gtIm(xx+1,1)],[gtIm(xx,2),gtIm(xx+1,2)],'color','b','linewidth',2)
%                             end
%                         end
                       % err = mean(sqrt(sum((gp(9:end,:) - ph(9:end,:)).^2,2)));
                        fprintf('Vanilla Vislet LSTM Error: %d \n',err);
                        pause(0.8)
                    end
                end
                batchCounter=batchCounter+1;
            end
        end
    end
end
mean(meanErr)
% mean(social_vislet_lstm_mat(:,2))
% save(fileSaveName,'social_vislet_lstm_mat')