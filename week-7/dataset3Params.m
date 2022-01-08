function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1; %[0:01; 0:03; 0:1; 0:3; 1; 3; 10; 30];
sigma = 0.3; %[0:01; 0:03; 0:1; 0:3; 1; 3; 10; 30];
size = 8;
% 0:01; 0:03; 0:1; 0:3; 1; 3; 10; 30
test_val = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

r = zeros(64, 3);
temp = 0;
for i = 1:8
  C = test_val(i);
  for j = 1:8
    sigma = test_val(j);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    r(++temp, :) = [mean(double(predictions ~= yval)) C sigma];
  endfor

end
[x ix] = min(r(:,1)); 
C = r(ix, 2);
sigma = r(ix, 3);



% =========================================================================

end
