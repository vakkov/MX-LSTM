function [gtPts,thisDt,frameId,pedId,thisPed,frameInfo,linInd]=extractNomralizedTraj(data,ii)
thisGt=data{ii,1};
xMatGT=thisGt(:,:,2);
yMatGT=thisGt(:,:,3);

thisDt=data{ii,2};
xMatDT=thisDt(:,:,2);
yMatDT=thisDt(:,:,3);


infoPed=data{ii,4};
pedId=infoPed(:,:,1);
frameInfo=infoPed(:,:,2);

thisPedID=data{ii,5}(4,:)'; % extract only pedetsrian id
myPed=unique(thisPedID);
thisPed=myPed;

[r,c]=find(pedId==thisPed);
linInd=sub2ind(size(pedId),r,c);

gtPts(:,1)=xMatGT(linInd);
gtPts(:,2)=yMatGT(linInd);
thisDt(:,1)=xMatDT(linInd);
thisDt(:,2)=yMatDT(linInd);
frameId=frameInfo(linInd);

end