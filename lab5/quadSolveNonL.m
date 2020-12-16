function [alpha, b] = quadSolveNonL(X, y, gamma)
m = size(X, 1);
C = 1;  % 0.000001, 0.00001,..., 0.1...
%% solve the QP problem
%H = (X*X') .* (y*y');
H = zeros(m,m);
for i=1:m
	for j=1:m
		H(i,j) = y(i) * y(j) * exp(-gamma * sum((X(i,:)-X(j,:)).^2) );
	end
end

f = -ones(m, 1);

Aeq = y';
beq = 0;
lb = zeros(m, 1);
ub = C * ones(m,1);
A = [];
b_q = [];

alpha = quadprog(H,f,A,b_q,Aeq,beq,lb,ub);

%% get the result
epsilon = 1e-6;  % control the precision!!!!!!  -> 0=> wrong!
y_j_idx = find(alpha>epsilon & alpha<C);
y_j = y(y_j_idx(1));

%w = X' * (alpha .* y);
mins = 0;
for i=1:m
	mins = mins + alpha(i) * y(i) * exp(-gamma * sum((X(i,:) - X(y_j_idx(1),:)).^2) );
end;
b = y_j - mins;

end

