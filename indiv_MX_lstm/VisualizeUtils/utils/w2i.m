function pts=w2i(d,H)

persons=1;
for i = 1:length(persons)
 %   p{i} = seq.obsmat(seq.obsmat(:,2) == persons(i),[3 5]);
    p{i} = d;
    p{i} = [p{i} ones(size(p{i},1),1)] / H';
    p{i} = round(p{i}(:,1:2) ./ repmat(p{i}(:,3),[1 2]));
end
pts=p{i};
pts=abs(pts);
% imshow(img), hold on
% plot(pts(:,2),pts(:,1) ,'r*')
