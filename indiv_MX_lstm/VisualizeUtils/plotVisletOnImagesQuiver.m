function plotVisletOnImagesQuiver(startingPointX,startingPointY,xMax2,yMax2,color)
p1=[startingPointX(:) startingPointY(:)];
p2=[xMax2(:) yMax2(:)];
dp=p2-p1;
for ii=1:length(xMax2)
    quiver(startingPointX(ii), startingPointY(ii),dp(ii,1),dp(ii,2),color,'lineWidth',1,'MaxHeadSize',200,'autoScale','off');
  %  set(K,'AutoScaleFactor',0);
end





end