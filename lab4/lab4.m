close all; clc, clear;


%% preparations
%======================================================================================
train_data = load("training_data.txt");
X = train_data(:, 1:8);
y = train_data(:, 9);  % admission decision recommendation

test_data = load("test_data.txt");
X_test = test_data(:, 1:8);
y_test = test_data(:, 9);

m = size(X, 1);
n = 8;

y_res = zeros(size(y_test,1), 1);
num_features = [3, 5, 4, 4, 3, 2, 3, 3];  % laplace smoothing

% accuracy test
idx_ran = randperm(m);
len = [500:500:10000];  % size of smaller training set


%% calculation
%======================================================================================
acr = zeros(size(len));
for acr_test=1:length(len)
	% different size of training set
	X_train = X( idx_ran(1:len(acr_test)), :);
	y_train = y( idx_ran(1:len(acr_test)), :);
	m = size(X_train, 1);

	% apply naive bayes
	run = size(X_test, 1);
	for i=1:run
		x_i = X_test(i, :);  % i-th test data
		p = 0; final_class = 0;
		for class=0:4
			idx = find(y_train==class);
			%% p(y)
			p_y = (size(idx, 1) +1) / (m+5);  % laplace smoothing

			%% p(x|y)
			for j=1:n
				x_class = X_train(idx, :);
				idx_tmp = find(x_class(:, j)==x_i(j));
				p_y = p_y * ( (size(idx_tmp, 1) +1) / (size(idx, 1) +num_features(j)) );
			end;

			%% refresh p
			if p_y > p
				p = p_y; 
				final_class = class;
			end;
		end;
		y_res(i) = final_class;
	end;

	arr_true = (y_res==y_test);
	accuracy = size(find(arr_true==1), 1) / run;  % accuracy
	acr(acr_test) = accuracy;
end;

plot(len, acr, 'b-o');  % validate the accuracy with sizes of training set

