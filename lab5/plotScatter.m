function [] = plotScatter(X, y)

X_idx1 = find(y==1);
X_idx2 = find(y==-1);
X_pos = X(X_idx1,:);
X_neg = X(X_idx2,:);
figure;
plot(X_pos(:,1), X_pos(:,2), 'r+');hold on
plot(X_neg(:,1), X_neg(:,2), 'bo');

end