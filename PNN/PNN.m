% Irwan Winarto
% References: A First Course in Machine Learning, Chapter 2.
%% gauss_surf.m

clc;clear all;close all;
%% The Multi-variate Gaussian pdf is given by:
% $p(\mathbf{x}|\mu,\Sigma) =
% \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left\{-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right\}$

% Read iris.csv. Our input will be coordinates (x, y), we will just use the Sepal Length (x) and Petal Length (y).
% iris setosa: row 1-50
% iris-versicolor: row 51-101
% iris-virginica: row 102-150

% read data
data = readtable('iris.csv')

% Get columns
sl = data(:,2);
% convert numeric table -> array
sl = table2array(sl);

% repeat
pl = data(:,4);
pl = table2array(pl);

% divide the data among 3 classes
tmp = [sl pl];
data1 = tmp(1:50, :);
data2 = tmp(51:101,:);
data3 = tmp(102:150,:);

% graph limits
xmax = 7;
xmin = 0;
ymax = xmax;
ymin = xmin;

%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------
% CLASS 1 Multivariate Gaussian - IRIS-SETOSA
%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------

%data1 = randn(5,2)+3
pdfv1 = 0;
n = 1;

for i = 1:size(data1, 1)
    %% Define the Gaussian
    %mu = [1;2];
    mu = [data1(i,1) data1(i,2)];
    sigma = [1 0.8;0.8 4];
    %% Define the grid for visualisation
    x=[xmin:0.1:xmax];
    y=[ymin:0.1:ymax];
    [X,Y] = meshgrid(x,y);
    %% Define the constant
    const = (1/sqrt(2*pi))^2;
    const = const./sqrt(det(sigma));
    temp = [X(:)-mu(1) Y(:)-mu(2)];
    pdfv = const*exp(-0.5*diag(temp/(sigma)*temp'));
    pdfv = reshape(pdfv,size(X));
    pdfv1 = pdfv1 + const*exp(-0.5*diag(temp/(sigma)*temp'));
    
    %% Make the plots
%    figure(1);hold off
%    contour(X,Y,pdfv);

%     figure(n)
%     n = n+1;
%     
%     surf(X,Y,pdfv);
%     title(['Multivariate Gaussian of data #', num2str(n)]);
end
figure(n)

pdfv1 = reshape(pdfv1,size(X));
surf(X,Y,pdfv1);
title('All the Class 1 Multivariate Gaussians added up');

%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------
% CLASS 2 Multivariate Gaussian - IRIS-VERSICOLOR
%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------

cut = n;
n = n+1;

%data2 = randn(10,2)+5
pdfv2 = 0;

for i = 1:size(data2, 1)
    %% Define the Gaussian
    %mu = [1;2];
    mu = [data2(i,1) data2(i,2)];
    sigma = [1 0.1;0.6 3];
    %% Define the grid for visualisation
    x=[xmin:0.1:xmax];
    y=[ymin:0.1:ymax];
    [X,Y] = meshgrid(x,y);
    %% Define the constant
    const = (1/sqrt(2*pi))^2;
    const = const./sqrt(det(sigma));
    temp = [X(:)-mu(1) Y(:)-mu(2)];
    pdfv = const*exp(-0.5*diag(temp/(sigma)*temp'));
    pdfv = reshape(pdfv,size(X));
    pdfv2 = pdfv2 + const*exp(-0.5*diag(temp/(sigma)*temp'));
    
    %% Make the plots
%    figure(1);hold off
%    contour(X,Y,pdfv);

%     figure(n)
%     n = n+1;
%     
%     surf(X,Y,pdfv);
%     title(['Class 2 Multivariate Gaussian of data #', num2str(n-cut)]);
end
figure(n)
pdfv2 = reshape(pdfv2,size(X));
surf(X,Y,pdfv2);
title('All the Class 2 Multivariate Gaussians added up');

%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------
% CLASS 3 Multivariate Gaussian - IRIS-VIRGINICA
%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------

cut = n;
n = n+1;

%data2 = randn(10,2)+5
pdfv3 = 0;

for i = 1:size(data3, 1)
    %% Define the Gaussian
    %mu = [1;2];
    mu = [data3(i,1) data3(i,2)];
    sigma = [1 0.1;0.6 3];
    %% Define the grid for visualisation
    x=[xmin:0.1:xmax];
    y=[ymin:0.1:ymax];
    [X,Y] = meshgrid(x,y);
    %% Define the constant
    const = (1/sqrt(2*pi))^2;
    const = const./sqrt(det(sigma));
    temp = [X(:)-mu(1) Y(:)-mu(2)];
    pdfv = const*exp(-0.5*diag(temp/(sigma)*temp'));
    pdfv = reshape(pdfv,size(X));
    pdfv3 = pdfv3 + const*exp(-0.5*diag(temp/(sigma)*temp'));
    
    %% Make the plots
%    figure(1);hold off
%    contour(X,Y,pdfv);

%     figure(n)
%     n = n+1;
%     
%     surf(X,Y,pdfv);
%     title(['Class 2 Multivariate Gaussian of data #', num2str(n-cut)]);
end
figure(n)
pdfv3 = reshape(pdfv3,size(X));
surf(X,Y,pdfv3);
title('All the Class 3 Multivariate Gaussians added up');

%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------
% Adding both totals
% adding pdfv1 and pdfv2 together.
%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------

n = n + 1;
pdfv_sum = pdfv1 + pdfv2 + pdfv3;
figure(n)
surf(X, Y, pdfv_sum);
title('Combination');

%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------
% Classifying new inputs
% input is xin and yin = (xin, yin). The z-value is found by calculating
% the row and column number based on the coordinates inputted.
%______________________________________________________________________________________________
%----------------------------------------------------------------------------------------------

% using (x, y) = (6, 6) because it is the maximum point being plotted.
% (6,6) should be the 111th row and column of pdfv3
xin = 6;
yin = 6;
size(X,1)
xloc = round(xin / xmax * size(X,1))
yloc = round(yin / ymax * size(Y,1))

%Prob = pdfv3(xloc, yloc)
prob_class1 = pdfv1(xloc, yloc)
prob_class2 = pdfv2(xloc, yloc)
prob_class3 = pdfv3(xloc, yloc)

%if(prob_class1 > prob_class2 && prob_class1 > prob_class3)
 winner = max([prob_class1, prob_class2, prob_class3])
 if prob_class1 == winner
     fprintf('It is IRIS-VIRGINICA .\n')
 elseif prob_class2 == winner
     fprintf('It is IRIS-VERSICOLOR.\n')
 elseif prob_class3 == winner
     fprintf('It is IRIS-SETOSA.\n')
 end
