close all; clc, clear;

[X_train, y_train] = genFeatures("train-01-images.svm");
[X_test, y_test] = genFeatures("test-01-images.svm");

[w,b] = quadSolve(X_train, y_train);

%
y_pred_train = sign(X_train * w + b);

true_idx1 = (y_pred_train==y_train);
accuracy1 = size(find(true_idx1==1), 1) / size(y_train, 1);

%
y_pred_test = sign(X_test * w + b);

true_idx2 = (y_pred_test==y_test);
accuracy2 = size(find(true_idx2==1), 1) / size(y_test, 1);
