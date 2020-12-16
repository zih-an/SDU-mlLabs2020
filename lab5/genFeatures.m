function [X, y] = genFeatures(filename)
%% load the file
fidin = fopen(filename); % 打开test2.txt文件 
n = 1;
apres = [];

while ~feof(fidin)
  tline = fgetl(fidin); % 从文件读行 
  apres{n} = tline;
  n = n+1;
end

%% matrix
X = [];
y = [];

% generate
for i=1:n-1
  a = char(apres(i));  % 第n个数字
  % get the class
  num = sscanf(a(1:3), "%d");
  y = [y; num];

  % get the features
  lena = size(a);
  lena = lena(2);  % 取得第a行的列数
  xy = sscanf(a(4:lena), '%d:%d');

  lenxy = size(xy);
  lenxy = lenxy(1);
  
  % grid features 28*28=784
  grid = [];
  grid(784) = 0;
  for i=2:2:lenxy  %% 隔一个数
      if(xy(i)<=0)
          break
      end
    grid(xy(i-1)) = xy(i) * 100/255;
  end
  grid1 = reshape(grid,28,28);
  grid1 = fliplr(diag(ones(28,1)))*grid1;
  grid1 = rot90(grid1,3);
  
  %image(grid1)
  %hold on;
  features = extractLBPFeatures(grid1);  % get the LBP feature
  X = [X; features];  % build the feature matrix
end

X = double(X);
y = double(y);
end
