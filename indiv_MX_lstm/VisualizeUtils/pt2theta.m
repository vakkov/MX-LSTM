function theta=pt2theta(pt1,pts2)


theta=atan2d((pts2(2)-pt1(2)),(pts2(1)-pt1(1)));
if(theta<0)
   theta=theta+360; 
end


end