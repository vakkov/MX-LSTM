addpath('/export/work/i.hasan/datasets/frames_zara01/')
load('D_zara01.mat')

img=imread(sprintf('%06d.png',1));
imshow(img), hold on

[x,y]=getpts();
worldCords=i2w([x(:) y(:)],D.H);
err = mean(sqrt(sum((worldCords(1,:) - worldCords(2,:)).^2,2)))



