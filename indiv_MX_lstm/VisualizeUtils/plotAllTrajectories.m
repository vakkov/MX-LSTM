
addpath('dataFiles')
addpath('utils')
addpath('output')
addpath('alteredTrajFiles')
allSets={'zara01','zara02','stu03'};
datasets=allSets{1};
if strcmp(datasets,'zara01')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara01\')
    %% name of the output file below by running test_pedestrian_wise
    load('z1_traj_struct_in_correct_vislet_basleine.mat')
    load('z1_traj_struct_in_correct_vislet_east.mat')
    
elseif strcmp(datasets,'zara02')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara02\')
    %% name of the output file below by running test_pedestrian_wise
    load('z2_traj_struct_correct_vislet.mat')
    load('z2_traj_struct_in_correct_vislet.mat')
    
elseif strcmp(datasets,'stu03')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_stu03\')
    load('ucy_traj_struct_correct_vislet.mat')
    load('ucy_traj_struct_in_correct_vislet.mat')
    
end
plotVislet=1;

obcol = [237,125,49] / 255 ;
thisFrame=337;
allFrame=vertcat(incorrectData(1:end).frameId);
allPeds=vertcat(incorrectData(1:end).pedId);
sampIntervalPed=vertcat(incorrectData(1:end).intervalPedId);
sampIntervalFrame=vertcat(incorrectData(1:end).intervalStartingFrame);
im=imread(sprintf('%06d.png',thisFrame));
figure,
imshow(im), hold on
msize=5;
idx=find(thisFrame==allFrame);
pedThisFrame=allPeds(idx);
pedThisFrame=unique(pedThisFrame);
for ii=1:length(pedThisFrame)
    thisPed=pedThisFrame(ii);
    allIntervals=find(thisPed==sampIntervalPed);
    intervalFrames=sampIntervalFrame(allIntervals);
    difVect=abs(intervalFrames-thisFrame);
    [rv,ri]=min(difVect);
   % structId=find((sampIntervalFrame==rv & sampIntervalPed==thisPed));
    gtIm=incorrectData(allIntervals(ri)).gtIm;
    pimflase=incorrectData(allIntervals(ri)).predIm;
    pimTrue=correctData(allIntervals(ri)).predIm;
    
    gtImVis=correctData(allIntervals(ri)).gtImVis;
    gtImVisFalse=incorrectData(allIntervals(ri)).gtImVis;
    pimflaseVis=incorrectData(allIntervals(ri)).predImVis;
    pimTrueVis=correctData(allIntervals(ri)).predImVis;
    
    for xx=1:size(gtIm,1)-1
        
        if xx>=8
   
            line([pimflase(xx,1) pimflase(xx+1,1)],[pimflase(xx,2) pimflase(xx+1,2)],'color','y','linewidth',2)
%             plotVisletOnImagesQuiver5(pimflase(xx,1),pimflase(xx,2),pimflaseVis(xx,1),pimflaseVis(xx,2),'r',50)
            
          %  line([pimTrue(xx,1) pimTrue(xx+1,1)],[pimTrue(xx,2) pimTrue(xx+1,2)],'color','b','linewidth',2)
%            line([gtIm(xx,1) gtIm(xx+1,1)],[gtIm(xx,2) gtIm(xx+1,2)],'color','g','linewidth',2)
        else
            line([gtIm(xx,1) gtIm(xx+1,1)],[gtIm(xx,2) gtIm(xx+1,2)],'color','k','linewidth',2)
%            plotVisletOnImagesQuiver5(gtIm(xx,1),gtIm(xx,2),gtImVis(xx,1),gtImVis(xx,2),'k',50)
%             plotVisletOnImagesQuiver5(gtIm(xx,1),gtIm(xx,2),gtImVisFalse(xx,1),gtImVisFalse(xx,2),'y',50)
        end
    end
    plot(pimflase(9:end,1),pimflase(9:end,2), 'o','MarkerSize',msize,'MarkerEdgeColor',obcol,'MarkerFaceColor',obcol)
    plot(gtImVisFalse(:,1),gtImVisFalse(:,2),'r*')
    
end
asd=1;



