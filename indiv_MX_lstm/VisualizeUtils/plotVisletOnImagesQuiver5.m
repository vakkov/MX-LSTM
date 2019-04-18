%% author: irtiza
%% input: StartingPointX   size:[1x1]
%% input: StartingPointY   size:[1x1]
%% input: angleInDegree    size:[1x1]
%% output: Visualizations of the angle on image/figure

function plotVisletOnImagesQuiver5(startingPointX,startingPointY,xMax2,yMax2,color,fl)
p1=[startingPointX(:) startingPointY(:)];
p2=[xMax2(:) yMax2(:)];

% p2 = p1 + fl*(p2-p1)/norm(p2-p1);

q1 = sqrt(sum((p2-p1).^2,2)) ;
p2 = p1 + fl* bsxfun( @rdivide, (p2-p1), q1) ;
% gl;
% %MYstep
% % gl
% MYstep=0;
% while norm(p2-p1)>fl
%     MYstep=MYstep+1;
%     p2 = p1 + MYstep*(p2-p1);
%     %gl2=norm(p2-p1);
% end


for ii=1:length(xMax2)
   davinci( 'arrow', 'X',[p1(ii,1),p2(ii,1)] , 'Y',  [p1(ii,2),p2(ii,2)],'Color',color,'Head.Width',20,'Head.Length',5,'LineWidth',1.5);
end

end