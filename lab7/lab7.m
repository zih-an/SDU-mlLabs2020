close all;  clear, clc;

TRAIN = [];
TEST = [];

train_size = 7;
K = 50;

%%===========================open the images============================
for i=1:40
	idx = randperm(10);
	for j=1:10
		f = idx(j);  %the image file number
		filename = sprintf("orl_faces/s%d/%d.pgm", i, f);
		img = double(imread(filename));
		img = reshape(img, 1, size(img,1)*size(img,2));
		% split
		if j<=train_size
			TRAIN = [TRAIN; img];
		else
			TEST = [TEST; img];
		end
	end
end


%%==========================implement pca===============================
%1. center the data
TRAIN = TRAIN - mean(TRAIN, 2);
TEST = TEST - mean(TEST, 2);
%2. covariance matrix
S = (TRAIN' * TRAIN) / size(TRAIN,1);
%3. eigen decomposition
[V, D] = eig(S); 
%descending
[D, index] = sort(diag(D),'descend');
V = V(:,index);
%4. take the first K
U = V(:, 1:K);
%5. projection
Z = TRAIN * U;
Zt= TEST * U;


%%=========================SVM: 1 vs. all===============================
C = 1;
ws = [];
bs = [];
for i=1:40
	y_train = -1 * ones(size(Z,1), 1);
	beg = (i-1)*7+1;
	for j=beg:beg+6
		y_train(j) = 1;
	end
	[w,b] = quadSolve(Z, y_train, C);
	ws = [ws w];
	bs = [bs b];
end


%%========================make predictions=============================
RESVAL = Zt * ws + bs;
[val labels] = max(RESVAL');
y_pred = labels';
y_test = [];
for i=1:size(y_pred)
	y_test = [y_test; floor((i-1)/3)+1];
end;

right_idx = (y_pred==y_test);
accuracy = size(find(right_idx==1), 1) / size(y_pred, 1);
y_con = [y_pred y_test];

