function pts=i2w(locs,H)
    P = [locs ones(size(locs,1),1)] * H';
    pts = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
end