%% (n-1xframeRate)+1;

function frId=refFrame2origFrame(ofr,nfr,frc)
% 
for ii=1:length(frc)
thisFrame=frc(ii);
idx=find(nfr==thisFrame);
frId(ii,1)=ofr(idx);
end



% 
% for ii=1:length(fr)
% thisFrame=fr(ii);
% origId=((thisFrame-1)*df)+1;
% frId(ii,1)=origId;
% end




end