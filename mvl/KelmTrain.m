function [Htrain,inW, bias, outW, scores] = KelmTrain( alpha, X, Y, Opt)
% FUNCTION trains the Extreme Learning Machine. The activation function is
% sigmoid which could be changed easily if needed.
%
%	[inW, bias, outW, scores] = elmTrain( X, Y, nHiddenNeurons, C );
%
% INPUT :
%	X				- data patterns (column vectors)
%	Y				- numeric labels for each pattern (1, ... )
%	nHiddenNeurons	- number of hidden neurons
%	C				- regularization parameter
%
% OUTPUT :
%	inW				- input weights matrix
%	bias			- bias vector
%	outW			- output weights matrix
%	scores			- scores on the own training data
%
kernel_type=Opt.kernel_type;
kernel_para=Opt.kernel_para;
nHiddenNeurons=Opt.nHiddenNeurons;
C=Opt.C;
% lambda=Opt.lambda;
nTrainData = size( X, 1 );
nInputNeurons = size( X, 2 );

%nClasses = length( unique(Y) );
%nClasses = size(Y,1);

%targets = zeros( nClasses, nTrainData );

% populate 1-of-k target matrix - transforming labels vector
% for i = 1 : nTrainData
% 	targets( Y(i), i ) = 1;
% end
% targets = targets * 2 - 1;	% from 0/1 to -1/1

% generate random input weight matrix
inW = rand( nInputNeurons, nHiddenNeurons ) * 2 - 1;

% generate random hidden neuron vector
bias = rand( nHiddenNeurons, 1 );

% compute the pre-H matrix
preH = X * inW ;

% build the bias matrix
biasM = repmat( bias, 1, nTrainData );

% update the H matrix
preH = preH + biasM';

% calculate hidden neuron output matrix H
% Htrain = 1 ./ (1 + exp(-preH));
% Htrain = 2 ./ (1 + exp(-2*preH)) - 1;
% Htrain = preH;
switch (Opt.fun)
    case 'sigmoid'
        Htrain = 1 ./ (1 + exp(-preH));
    case 'tanh'
        Htrain = 2 ./ (1 + exp(-2*preH)) - 1;
    case 'purelin'
        Htrain = preH;
    case 'elliotsig'
        Htrain = preH./(1+abs(preH));
    case 'relu'
        Htrain = max(preH,0);
end
% Htrain = tanh(preH);
% compute regularized output weights matrix
% Y1=Y'*T_4;
Ktrain = kernel_matrix(Htrain,kernel_type, kernel_para);
% outW = ( eye(nHiddenNeurons)/C + Ktrain + lambda*L*Ktrain) \( Y);
outW = ( eye(nTrainData)/C + alpha*Ktrain ) \(alpha*Y);
% output for the training data
scores = (Ktrain * outW);
end
