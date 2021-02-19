% -- Load Movie
load cam3_2.mat;
[height width rgb num_frames] = size(vidFrames3_2);
% -- Watch Movie
for j=1:num_frames
X=vidFrames3_2(:,:,:,j);
imshow(X); drawnow
end