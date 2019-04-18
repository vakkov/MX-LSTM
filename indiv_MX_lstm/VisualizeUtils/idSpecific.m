

addpath('dataFiles')
addpath('utils')
addpath('output')
addpath('alteredTrajFiles')
allSets={'zara01','zara02','stu03'};
datasets=allSets{2};
if strcmp(datasets,'zara01')
    addpath('C:\Users\i.hasan-ext\Documents\Trajectory Forecasting\cvpr2011\pedestrians\pedestrians\frames_zara01\')
    %% name of the output file below by running test_pedestrian_wise
    load('z1_traj_struct_correct_vislet.mat')
    load('z1_traj_struct_in_correct_vislet.mat')
    
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

genVisualization=1;
offSet=10;
im=imread(sprintf('%06d.png',140));
imshow(im), hold on
for ii=7:size(correctData,2)
    if ii==7 || ii==14 
    if genVisualization
        
      %  figure
        
        for xx=1:size(correctData(ii).gtIm,1)-1
            line([correctData(ii).gtIm(xx,1) correctData(ii).gtIm(xx+1,1)],[correctData(ii).gtIm(xx,2) correctData(ii).gtIm(xx+1,2)],'color','g','linewidth',2)
         %   plotVisletOnImagesQuiver5(incorrectData(ii).gtIm(xx,1),incorrectData(ii).gtIm(xx,2),correctData(ii).gtImVis(xx,1),correctData(ii).gtImVis(xx,2),'m',50)
        end
        
        for xx=1:8
            line([correctData(ii).gtIm(xx,1) correctData(ii).gtIm(xx+1,1)],[correctData(ii).gtIm(xx,2) correctData(ii).gtIm(xx+1,2)],'color','g','linewidth',2)
            
            
            if xx==1
                theta=pt2theta([incorrectData(ii).gtIm(xx+1,1),incorrectData(ii).gtIm(xx+1,2)],[incorrectData(ii).gtImVis(xx+1,1),incorrectData(ii).gtImVis(xx+1,2)]);
                vislets=pts2vislet([incorrectData(ii).gtIm(xx,1),incorrectData(ii).gtIm(xx,2)],theta,offSet);
            %    plotVisletOnImagesQuiver5(incorrectData(ii).gtIm(1,1),incorrectData(ii).gtIm(1,2),vislets(1),vislets(2),'w',50)
                
            else
      %          plotVisletOnImagesQuiver(incorrectData(ii).gtIm(xx,1),incorrectData(ii).gtIm(xx,2),incorrectData(ii).gtImVis(xx,1),incorrectData(ii).gtImVis(xx,2),'w')
                theta=pt2theta([incorrectData(ii).gtIm(xx,1),incorrectData(ii).gtIm(xx,2)],[incorrectData(ii).gtImVis(xx,1),incorrectData(ii).gtImVis(xx,2)]);
                vislets=pts2vislet([incorrectData(ii).gtIm(xx,1),incorrectData(ii).gtIm(xx,2)],theta,offSet);
               % plotVisletOnImagesQuiver5(incorrectData(ii).gtIm(xx,1),incorrectData(ii).gtIm(xx,2),vislets(1),vislets(2),'w',50)
            end
        end
        predInt=correctData(ii).predIm(9:end,:);
        predIntC=incorrectData(ii).predIm(9:end,:);
  
%         predImVis=predImVis(9:end,:);
        for xx=1:size(predInt,1)-1
        %    line([predInt(xx,1) predInt(xx+1,1)],[predInt(xx,2) predInt(xx+1,2)],'color','b','linewidth',2)
            line([predIntC(xx,1) predIntC(xx+1,1)],[predIntC(xx,2) predIntC(xx+1,2)],'color','y','linewidth',2)
            
            theta=pt2theta([predIntC(xx,1),predIntC(xx,2)],[incorrectData(ii).predImVis(9+xx,1),incorrectData(ii).predImVis(9+xx,2)]);
            vislets=pts2vislet([predIntC(xx,1),predIntC(xx,2)],theta,offSet);
%             plotVisletOnImagesQuiver5(predIntC(xx,1),predIntC(xx,2),vislets(1),vislets(2),'r',50)
%             plotVisletOnImagesQuiver5(predInt(xx,1),predInt(xx,2),correctData(ii).predImVis(9+xx,1),correctData(ii).predImVis(9+xx,2),'k',50)
        end
    end
    
    end
end