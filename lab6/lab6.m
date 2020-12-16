close all; clear,clc;


BIRD = double(imread("bird_small.tiff"));
NEWBIRD = BIRD;
CLASS = zeros(128, 128);

% randomly pick 16 centoids
mu = zeros(16,3);
idx = randi([0,128],16,2);
for i=1:16
	mu(i,1) = BIRD(idx(i,1), idx(i,2), 1);
	mu(i,2) = BIRD(idx(i,1), idx(i,2), 2);
	mu(i,3) = BIRD(idx(i,1), idx(i,2), 3);
end

% algorithm
for iter=1:100
	new_mu = zeros(16,3);
	cnt = zeros(16,1);
	% picking the nearest centroids
	for i=1:128
		for j=1:128
			minDist=2000000; minDi=0;  %distance; centroid's index
			for di=1:16
				d = (BIRD(i,j,1)-mu(di,1))^2 + (BIRD(i,j,2)-mu(di,2))^2 + (BIRD(i,j,3)-mu(di,3))^2;
				%d = sqrt(d);
				if d<minDist
					minDist = d;
					minDi = di;
				end
			end
			% store the sum
			for dmns=1:3
				new_mu(minDi,dmns) = new_mu(minDi,dmns) + BIRD(i,j,dmns);
			end
			cnt(minDi) = cnt(minDi) + 1;
			CLASS(i,j) = minDi;
		end
	end
	% get new centroids
	for i=1:16
		% no point is assigned to i
		if cnt(i)==0
			for dmns=1:3
				new_mu(i,dmns) = mu(i,dmns);
			end
			continue
		end
		% get mean to update mu
		for dmns=1:3
			new_mu(i,dmns) = new_mu(i,dmns) / cnt(i);
		end
	end
	mu = new_mu;
end


%%========================small bird===========================
% get new bird
for i=1:128
	for j=1:128
		for dmns=1:3
			NEWBIRD(i,j,dmns) = new_mu(CLASS(i,j), dmns);
		end
	end
end

figure
imshow(uint8(round(double(BIRD))));
figure
imshow(uint8(round(NEWBIRD)));

%%========================large bird===========================
BIGBIRD = double(imread("bird_large.tiff"));
NEW_BIGBIRD = zeros(size(BIGBIRD));

for i=1:size(BIGBIRD,1)
	for j=1:size(BIGBIRD,2)
		minDist=2000000; minDi=0;  %distance; centroid's index
		for di=1:16
			d = (BIGBIRD(i,j,1)-mu(di,1))^2 + (BIGBIRD(i,j,2)-mu(di,2))^2 + (BIGBIRD(i,j,3)-mu(di,3))^2;
			%d = sqrt(d);
			if d<minDist
				minDist = d;
				minDi = di;
			end
		end
		% replace color
		for dmns=1:3
			NEW_BIGBIRD(i,j,dmns) = new_mu(minDi, dmns);
		end
	end
end

figure
imshow(uint8(round(double(BIGBIRD))));
figure
imshow(uint8(round(NEW_BIGBIRD)));

