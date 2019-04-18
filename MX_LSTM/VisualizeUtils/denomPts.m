function [gp,ph]=denomPts(gtPts,thisDt,normParams,allPedAnno,origFrames,normFrame,frameId)

exGt(:,1)=(gtPts(:,1)*normParams.sig(1))+normParams.mu(1);
exGt(:,2)=(gtPts(:,2)*normParams.sig(2))+normParams.mu(2);
exPred(:,1)=(thisDt(:,1)*normParams.sig(1))+normParams.mu(1);
exPred(:,2)=(thisDt(:,2)*normParams.sig(2))+normParams.mu(2);
frameId=refFrame2origFrame(origFrames,normFrame,frameId);

thisFrame=frameId(1);
tid=find(allPedAnno(:,1)==thisFrame);
tuple=allPedAnno(1:tid,[3 4]);
ptsTill=cumsum(tuple,1);
fristLoc=ptsTill(end,:);
exGt(1,:)=fristLoc;
exPred(1,:)=fristLoc;
gp=cumsum(exGt);
ph=cumsum(exPred);

end