allSets={'zara01','zara02','stu03'};
datasets=allSets{1};

if strcmp(datasets,'zara01')
    addpath('frames_zara01/')
    ucy_paths = { 'ucy_crowd/data_zara01/'};
    ucy_labels = {'zara01'};
    [ D, T, Obj ] = importDataWithOrientation(ucy_labels,ucy_paths);
    [newT,arrayT]=arrangeFrameWise(T);
    filename='alteredVisletsFiles/info_z2.csv';
elseif strcmp(datasets,'zara02')
    addpath('frames_zara02/')
    ucy_paths = { 'ucy_crowd/data_zara02/'};
    ucy_labels = {'zara02'};
    [ D, T, Obj ] = importDataWithOrientation(ucy_labels,ucy_paths);
    [newT,arrayT]=arrangeFrameWise(T);
    filename='info_z2.csv';
elseif strcmp(datasets,'stu03')
    addpath('frames_stu03/')
    ucy_paths = { 'ucy_crowd/data_students03/'};
    ucy_labels = {'student03'};
    [ D, T, Obj ] = importDataWithOrientation(ucy_labels,ucy_paths);
    [newT,arrayT]=arrangeFrameWise(T);
    filename='info_ucy.csv';
    
end
infoColumn=arrayT(:,2:5);

newt=zeros(size(arrayT,1),6);
infoLocs=zeros(size(arrayT,1),6);
newt(:,1:2)=arrayT(:,[2 3]);
infoLocs=arrayT(:,[2 3]);
locs=arrayT(:,[4 5]);
convLocs=w2i(locs,D.H);
theta=arrayT(:,end);

theta(:)=0;

offSet=10;





vislets=pts2vislet(convLocs,theta,offSet);
vislets=i2w(vislets,D.H);
infoColumn(:,5:6)=vislets;
csvwrite(filename,infoColumn);

allPed=unique(newt(:,2));
firstLocs=[];
for ii=1:length(allPed)
    thisPed=allPed(ii);
    idx=find(newt(:,2)==thisPed);
    derLocs=[];
    L0T=arrayT(idx(1),[4 5]);
    L0=[0 0];
    firstLocs(ii,2:3)=L0;
    firstLocs(ii,1)=thisPed;
    derLocs=diff(arrayT(idx,[4 5]));
    %% append on top the first location wrt you computed df/dx
    derLocs=[L0;derLocs];
%     myDerLocs=[L0T;]
    %% store them back and siwtch xy to yx
    newt(idx,3)=derLocs(:,2);
    newt(idx,4)=derLocs(:,1);
    
    %% repeat the same process for vislets
    derVislets=[];
    %L0_Vislet=vislets(idx(1),:);
    L0_Vislet=[0 0];
    derVislets=diff(vislets(idx,:));
    derVislets=[L0_Vislet;derVislets];
    %% store them back and siwtch xy to yx
    newt(idx,5)=derVislets(:,2);
    newt(idx,6)=derVislets(:,1);
    firstLocs(ii,4:5)=L0_Vislet;
  
end

deltaX=mean(newt(:,4));
deltaY=mean(newt(:,3));
sX=std(newt(:,4));
sY=std(newt(:,3));
newt(:,4)=(newt(:,4)-deltaX)/sX;
newt(:,3)=(newt(:,3)-deltaY)/sY;



deltaVX=mean(newt(:,6));
deltaVY=mean(newt(:,5));
sVX=std(newt(:,6));
sVY=std(newt(:,5));
newt(:,6)=(newt(:,6)-deltaVX)/sVX;
newt(:,5)=(newt(:,5)-deltaVY)/sVY;
csvwrite('alteredVisletsFiles/z1_norm_zero_zero_mean_std_vislet_0degree.csv',newt');
save('alteredVisletsFiles/z1_norm_zer_zero_firstLoc_0degree.mat','deltaX','deltaY','sX','sY','deltaVX','deltaVY','sVX','sVY','firstLocs')







