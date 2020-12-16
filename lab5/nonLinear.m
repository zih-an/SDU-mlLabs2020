close all; clear, clc;


%================================data3===================================
%%====load data3
training_data = load('training_3.txt');
% training data 1
X = training_data(:, 1:2);
y = training_data(:, 3);

m = size(X, 1);

%%====train data3
gammas = [1,10,100,1000];
for g=1:4
	gamma = gammas(g);
	[alpha, b] = quadSolveNonL(X, y, gamma);
	plotScatter(X, y);

	% plot decision boundary
	xplot = linspace(min(X(:, 1)), max(X(:, 1)), 100)';
	yplot = linspace(min(X(:, 2)), max(X(:, 2)), 100)';
	[XX, YY] = meshgrid(xplot, yplot);
	vals = zeros(size(XX));

	%
	for i=1:size(XX,1)
		for j=1:size(XX,2)
			v = [XX(i,j) YY(i,j)];
			fx_w = 0;
			% compute w...part
			for w=1:m
				fx_w = fx_w + alpha(w)*y(w)*exp(-gamma * sum((X(w,:) - v).^2) );
			end
			vals(i,j) = sign(fx_w + b);
		end
	end
	
	% Plot the SVM boundary
	colormap bone
	contour (XX,YY, vals, [0 0],'LineWidth', 2);
	title(['gamma=',num2str(gamma)]);
end
