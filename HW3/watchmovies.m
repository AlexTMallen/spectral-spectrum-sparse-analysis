

% % -- Load Movie
% load cam1_2.mat;
% [height, width, rgb, num_frames] = size(vidFrames1_2);
% num_frames
% % -- Watch Movie
% for j=1:90
% X=vidFrames1_2(:,:,:,j);
% imshow(X); drawnow
% end

% -- Load Movie
load cam2_4.mat;
[height, width, rgb, num_frames] = size(vidFrames2_4);
% -- Watch Movie

for j=1:10
X=vidFrames2_4(:,:,:,j);
imshow(X); drawnow
end

% 2 starts earlier than 1. 3 starts a little bit later than 1