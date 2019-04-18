%%

% Computing a point based on current location and the head orientation
%  
% INPUTS:
%  - pts: <Nx2> matrix of pedestrians' locations in the current frame
%  - theta: <Nx1> matrix of pedestrians' orientation(degree) in the current 
%   frame
%  - offSet: How far you want to extend thet pint with ref to curret
%  location
% 
% OUTPUTS:
%  - vislets: <Nx2> matrix of pedestrians' locations in based on
%  orientation
%
%
function vislets=pts2vislet(pts,theta,offSet)


xMax=offSet*cos(deg2rad(theta));
yMax=offSet*sin(deg2rad(theta));
vislets(:,1)=pts(:,1)+xMax;
vislets(:,2)=pts(:,2)+yMax;



end