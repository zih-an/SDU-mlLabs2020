function [w, b] = quadSolve(X, y, C)
m = size(X, 1);
%% solve the QP problem
H = (X*X') .* (y*y');
f = -ones(m, 1);

Aeq = y';
beq = 0;
lb = zeros(m, 1);
ub = C * ones(m,1);
A = [];
b_q = [];

alpha = quadprog(H,f,A,b_q,Aeq,beq,lb,ub);

%% get the result
epsilon = 1e-9;  % control the precision!!!!!!  -> 0=> wrong!
y_j_idx = find(alpha>epsilon & alpha<C);
y_j = y(y_j_idx(1));

w = X' * (alpha .* y);
b = y_j - (alpha .* y)' * (X * X(y_j_idx(1), :)');

end

