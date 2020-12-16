close all; clc,clear;


%% Linear Regression
%========================================================================
% Preparations
X_lin = load('ex3Linx.dat');
y_lin = load('ex3Liny.dat');
m = size(y_lin,1);

% plot y as a function of x
plot(X_lin,y_lin,'k-o'); hold on

% add features
X_lin = [ones(m,1), X_lin, X_lin.^2, X_lin.^3, X_lin.^4, X_lin.^5];
n = size(X_lin, 2);

lambda_lin_m = eye(n);
lambda_lin_m(1,1) = 0;
lambda = 0;

% plot the curve
x_lin_max = max(X_lin(:,2));
x_lin_min = min(X_lin(:,2));
x_lin_plot = x_lin_min:0.001:x_lin_max;
tmp = size(x_lin_plot,2);
x_lin_plot = x_lin_plot';
x_lin_plot = [ones(tmp,1), x_lin_plot, x_lin_plot.^2, x_lin_plot.^3, x_lin_plot.^4, x_lin_plot.^5];

for lambda=[0, 1, 10]
	% Nomal Equations
	theta = inv(X_lin' * X_lin + lambda * lambda_lin_m)*X_lin'*y_lin;
	if lambda==0
		plot(x_lin_plot(:,2), x_lin_plot*theta, 'r-'); hold on
	elseif lambda==1
		plot(x_lin_plot(:,2), x_lin_plot*theta, 'g-'); hold on	
	else
		plot(x_lin_plot(:,2), x_lin_plot*theta, 'b-'); 
	end;
end;

legend('line chart', 'lambda=0', 'lambda=1', 'lambda=10');



%% Logistic Regression
%========================================================================
% Preparations
X_log = load('ex3Logx.dat');
y_log = load('ex3Logy.dat');
m = size(X_log,1);

% plot the markers
%figure
%pos = find(y_log==1); neg = find(y_log==0);
%plot(X_log(pos,1), X_log(pos,2), '+'); hold on
%plot(X_log(neg,1), X_log(neg,2), 'o');

% map the features
X_log = map_feature(X_log(:,1), X_log(:,2));
n = size(X_log,2);

% parameters
lambda_log_m = eye(n);
lambda_log_m(1,1) = 0;

g = inline('1./(1+exp(-z))');  % logistic

norm_theta = [];
for lambda=[0, 1, 10]
	% Paremeters
	theta = zeros(n, 1);
	j_for = 0;
	epsilon = 10^(-6);
	iter = 1;
	J_new = zeros(100,1);

	% Newton's Method while loop
	while 1
		J = (-1./m)*( y_log'*log(g(X_log*theta)) + (1-y_log')*log(1-g(X_log*theta)) ) + (lambda/(2*m))*sum(sum(theta.^2));
		J_new(iter) = J;

		% calculate Hessian Matrix
		H = zeros(n,n);
		for i=1:m
			tmp = g(X_log(i,:)*theta);
			H = H + tmp*(1-tmp)*X_log(i,:)'*X_log(i,:);
		end;
		H = (1./m)*H + (lambda/m)*lambda_log_m;

		% refresh theta
		gradJ = X_log'*(g(X_log*theta)-y_log);
		gradJ(2:end) = gradJ(2:end) + lambda*theta(2:end);
		gradJ = (1./m)*gradJ;
		theta = theta - inv(H)*gradJ;
		
		if abs(j_for-J)<=epsilon
			break;
		end;
		j_for = J;
		iter = iter+1;
	end;

	% norm of theta
	norm_theta = [norm_theta norm(theta)];
	
	% plot the J of iter
	figure
	plot(0:iter-1, J_new(1:iter), 'r-');
	title(strcat('lambda=',num2str(lambda)));

	% plot the contour
	u = linspace(-1, 1.5, 200);
	v = linspace(-1, 1.5, 200);
	z = zeros(length(u), length(v));
	for i=1:length(u)
		for j=1:length(v)
			z(i,j) = map_feature(u(i), v(j)) * theta;
		end;
	end;
	z = z';
	figure
	pos = find(y_log==1); neg = find(y_log==0);
	plot(X_log(pos,2), X_log(pos,3), '+'); hold on
	plot(X_log(neg,2), X_log(neg,3), 'o'); hold on

	contour(u,v,z,[0,0],'LineWidth',2);
	title(strcat('lambda=',num2str(lambda)));

end;

