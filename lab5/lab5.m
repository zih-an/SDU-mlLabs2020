close all; clear, clc;


%================================data1===================================
%%====load data1
training_data1 = load('training_1.txt');
test_data1 = load('test_1.txt');
% training data 1
X_train_data1 = training_data1(:, 1:2);
y_train_data1 = training_data1(:, 3);
% test data 1
X_test_data1 = test_data1(:, 1:2);
y_test_data1 = test_data1(:, 3);

%%====train data1
[w,b] = quadSolve(X_train_data1, y_train_data1);
plotScatter(X_train_data1, y_train_data1);
%% plot the decision boundary
% w*x+b=0  //2=> w1*x1 + w2*x2 + b = 0  => x2 = -(w1*x1+b)/w2
x1_plot = X_train_data1(:, 1);
x2_plot = -(w(1)*x1_plot + b) ./ w(2);
plot(x1_plot, x2_plot, 'k-');


% plot the original test one's boundary+points
%plotScatter(X_test_data1, y_test_data1, w, b);
%% make prediction
y_pred1 = sign(X_test_data1 * w + b);
true_idx = (y_pred1==y_test_data1);
accuracy1 = size(find(true_idx==1), 1) / size(y_test_data1, 1);
% plot the predicted one
%plotScatter(X_test_data1, y_pred1, w, b);



%================================data2===================================
%%====load data2
training_data2 = load('training_2.txt');
test_data2 = load('test_2.txt');
% training data 1
X_train_data2 = training_data2(:, 1:2);
y_train_data2 = training_data2(:, 3);
% test data 1
X_test_data2 = test_data2(:, 1:2);
y_test_data2 = test_data2(:, 3);

%%====train data2
[w,b] = quadSolve(X_train_data2, y_train_data2);
plotScatter(X_train_data2, y_train_data2);
%% plot the decision boundary
% w*x+b=0  //2=> w1*x1 + w2*x2 + b = 0  => x2 = -(w1*x1+b)/w2
x1_plot = X_train_data2(:, 1);
x2_plot = -(w(1)*x1_plot + b) ./ w(2);
plot(x1_plot, x2_plot, 'k-');


% plot the original test one's boundary+points
%plotScatter(X_test_data2, y_test_data2, w, b);
%% make prediction
y_pred2 = sign(X_test_data2 * w + b);
true_idx = (y_pred2==y_test_data2);
accuracy2 = size(find(true_idx==1), 1) / size(y_test_data2, 1);
% plot the predicted one
%plotScatter(X_test_data2, y_pred2, w, b);

