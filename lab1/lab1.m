close all;
clc, clear;


%% 2D Linear Regression
% load the training set
X = load('ex1_1x.dat');
y = load('ex1_1y.dat');
% plot the training set
figure
plot(X, y, 'o');
ylabel('Height in meters');
xlabel('Age in years');

m = length(y);
n = size(X, 2);

X = [ones(m,1), X];

theta = zeros(n+1, 1);
alpha = 0.07;

% h_x = X * theta
% the theta_1 of the first iteration
theta_1 = zeros(n+1, 1);

% start gradient descent
for i=1:1500
	theta = theta - alpha * (1/m) * X' * (X*theta-y);
	if i==1
		theta_1=theta;
	end;	
end;

% plot the result
hold on
plot(X(:,2), X*theta, '-');
legend('Training data', 'Linear regression');

% predict the height of two boys of ages 3.5 and 7
X_pred_age = [3.5; 7];
X_pred_age = [ones(2,1), X_pred_age];

y_pred_height = X_pred_age * theta;



% J_theta = (1/(2*m)) * (X*theta - y)' * (X*theta - y);
% plot J_theta
J_vals = zeros(100, 100);
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
% theta0_vals = logspace(-3, 3, 100);
% theta1_vals = logspace(-1, 1, 100);
for i=1:length(theta0_vals)
	for j=1:length(theta1_vals)
		t = [theta0_vals(i); theta1_vals(j)];
		J_vals(i, j) = (1/(2*m)) * (X*t - y)' * (X*t - y);
	end;
end;

J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals);
xlabel('\theta_0'); ylabel('\theta_1');




%% Multivariate Linear Regression
X_multi = load('ex1_2x.dat');
y_multi = load('ex1_2y.dat');

m_multi = length(y_multi);
n_multi = size(X_multi, 2);

X_multi = [ones(m_multi,1), X_multi];

sigma = std(X_multi);
mu = mean(X_multi);

X_multi(:, 2:3) = (X_multi(:,2:3)-mu(2:3)) ./ sigma(2:3);

theta_multi = zeros(n_multi+1, 1);
alpha = 0.1;  % 0.001<=alpha<=10
J_multi = zeros(50, 1);

for i=1:50
	J_multi(i) = (1/(2*m_multi)) * (X_multi*theta_multi - y_multi)' * (X_multi*theta_multi - y_multi);
	theta_multi = theta_multi - alpha*(1/m_multi) * X_multi' * (X_multi*theta_multi - y_multi);
end;

figure;
plot(0:49, J_multi(1:50), 'b-');
xlabel('Num. of iterations');
ylabel('Cost J');

X_house = [1650, 3];
X_house = [ones(1,1), X_house];

X_house(:, 2:3) = (X_house(:,2:3)-mu(2:3)) ./ sigma(2:3);
y_price = X_house*theta_multi;

