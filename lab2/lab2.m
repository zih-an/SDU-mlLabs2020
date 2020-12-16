close all; clc,clear;

%% Preparations
%=========================================================================
% load the data
X = load('ex2x.dat');
y = load('ex2y.dat');

m = size(y,1);
X = [ones(m,1) X];  % x0 = 1
n = size(X,2);

% plot the data
pos = find(y==1); neg = find(y==0);
plot(X(pos,2), X(pos,3), '+'); hold on
plot(X(neg,2), X(neg,3), 'o');

% define sigmoid function using inline
g = inline('1.0 ./ (1.0 + exp(-z))');

% standarization
mu = mean(X);
sigma = std(X);
X_grad = X;
X_grad(:,2:3) = (X(:,2:3)-mu(2:3)) ./ sigma(2:3);


%% Gradient Descent
%==========================================================================
theta = zeros(n,1);
alpha = 0.1;
epsilon = 10^(-6);
L_for = 0;
iter = 1;
L = zeros(2000,1);

while 1
	L_now = (1./m) * ( -y'*log(g(X_grad*theta)) - (1-y)'*log(1-g(X_grad*theta)) );
	L(iter) = L_now;  % L(theta) each time
	theta = theta - alpha*(1./m)* (X_grad'*(g(X_grad*theta)-y));
	% stop condition
	if abs(L_now-L_for) <= epsilon
		break;
	end;
	L_for = L_now;
	iter = iter+1;
end;

%% plot the J(theta) with iterations
%figure;
%plot(0:(iter-1), L(1:iter), 'b-');

%% plot the decision boundary
max_value = max(X(:,2));
min_value = min(X(:,2));
X_p = min_value:0.001:max_value;
% ((X_p-mu(2))./sigma(2))
Y_p = -(theta(1,1) + theta(2,1) * ((X_p-mu(2))./sigma(2)) ) / theta(3,1);
% ( Y_p.*sigma(3) + mu(3) )
%plot(X_p, ( Y_p.*sigma(3) + mu(3) ), 'b-');

% make predictions
X_stu = [20, 80];
X_stu = [1, X_stu];
X_stu(:,2:3) = (X_stu(:,2:3)-mu(2:3)) ./ sigma(2:3);
y_stu = g(X_stu*theta);  % admitted
y_stu = 1-y_stu;  % probability that not be admitted


%% Newton's Method
%=========================================================================
theta_2 = zeros(n,1);
X_new = X_grad;
L_for = 0;
iter = 1;
L_new = zeros(100,1);


while 1
	L_now = (1./m) * ( -y'*log(g(X_new*theta_2)) - (1-y)'*log(1-g(X_new*theta_2)) );
	L_new(iter) = L_now;
	
	H = zeros(n,n);  % Hessian Matrix
	for i=1:m
		tmp = g(X_new(i,:)*theta_2) .* (1-g(X_new(i,:)*theta_2));
		H = H + tmp * X_new(i,:)'*X_new(i,:);
	end;
	
	H = (1./m) * H;
	theta_2 = theta_2 - inv(H) * ( (1./m)* (X_new'*(g(X_new*theta_2)-y)) );
	% stop condition
	if abs(L_now-L_for) <= epsilon
		break;
	end;
	L_for = L_now;
	iter = iter+1;
end;

%% plot the J(theta) with iterations
figure;
plot(0:(iter-1), L_new(1:iter), 'b-');



%% plot the decision boundary
% ((X_p-mu(2))./sigma(2))
Y_p = -(theta_2(1,1) + theta_2(2,1) * ((X_p-mu(2))./sigma(2)) ) / theta_2(3,1);
% ( Y_p.*sigma(3) + mu(3) )
%plot(X_p, ( Y_p.*sigma(3) + mu(3) ), 'r-');

% make predictions
y_stu_new = g(X_stu*theta_2);  % admitted
y_stu_new = 1-y_stu_new;  % probability that not be admitted










