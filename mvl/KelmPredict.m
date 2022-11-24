function scores = KelmPredict( alpha, X, Htrain,Opt,inW, bias, outW )
% FUNCTION predicts the labels of some testing data using a trained Extreme
% Learning Machine.
%
%	scores = elmPredict( X, inW, bias, outW );
%
% INPUT :
%	X				- data patterns (column vectors)
%	inW				- input weights vector (trained model)
%	bias			- bias vector (trained model)
%	outW			- output weights vector (trained model)
%
% OUTPUT :
%	scores			-output weights vector (trained model)
%
kernel_type=Opt.kernel_type;
kernel_para=Opt.kernel_para;
% number of test patterns
nTestData = size( X, 1 );

% compute the pre-H matrix
preH = X*inW ;

% build the bias matrix
biasM = repmat( bias, 1, nTestData );

% update the pre-H matrix
preH = preH + biasM';

% apply the activation function
switch (Opt.fun)
    case 'sigmoid'
        Htest = 1 ./ (1 + exp(-preH));
    case 'tanh'
        Htest = 2 ./ (1 + exp(-2*preH)) - 1;
    case 'purelin'
        Htest = preH;
    case 'elliotsig'
        Htest = preH./(1+abs(preH));
    case 'relu'
        Htest = max(preH,0);
end

Ktest = kernel_matrix(Htest,kernel_type,kernel_para,Htrain);
% compute prediction scores
scores = (alpha*Ktest * outW);
