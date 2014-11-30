function [labels,centroids,relError,iter,U_binarized] = emonmf(M,initClusts,MaxIters,initMethod)

% EM-onmf.
%
% <Input>
% 	M: (nonnegative) items-by-features matrix
%   initClusts: 
%   (if scalar) number of desired clusters; 
%   (if vector, of length "numItems") cluster of i-th item,1<=i<=numItems; 
%   (if matrix, of size "numClusters x numFeatures") matrix having 
%         numClusters row and numFeatures columns, each row vector
%         representing the centroid of j-th cluster, 1 <= j <= numClusters.
% initMethod: 
% 	0: random points extracted from dataset; 1: uniformly random
%
% <Output>
%  labels: vector of cluster assignment for each data point centroids:
%   	matrix of cluster centroids relError: sum of squared distances
%       /||M||_F^2; 
%   iter: number of iterations to convergence U_binazarized: for
%       image decompositions, to make a matrix from the vector of cluster
%       assigments.
%
%
% Example: [clusters,centroids,relError,actualIters] =
% emonmf(M,numClusters,maxIters);
%
%
%
% Written by Filippo Pompili
% (filippopompili.09@gmail.com)
%
% Reference paper: 
% F. Pompili, N. Gillis, P.-A. Absil, F. Glineur,
% "Two algorithms for orthogonal nonnegative matrix 
% factorization with application to clustering".
% Neurocomputing, Vol. 141, pp. 15-25, 2014.
%
%
% This code comes with no guarantee or warranty of any kind.
%

maxRestarts = 2; % if init centroids not supplied, 0 otherwise

if nargin < 3
    MaxIters = 100; % max iterations before restart occurs
end

[numItems,numFeatures] = size(M);
areInitCentroidsGiven = 0;
if length(initClusts) == 1
    numClusts = initClusts;    
    if nargin < 4
        initMethod = 0; % random points from dataset
    end
elseif size(initClusts,1) == numItems && size(initClusts,2) == 1
    numClusts = max(initClusts);
    labels = initClusts;
    centroids = updateCentroids(M,labels,numClusts);
    maxRestarts = 0;
elseif size(initClusts,1) > 1 && size(initClusts,1) <= numItems && size(initClusts,2) == numFeatures
    areInitCentroidsGiven = 1;
    numClusts = size(initClusts,1);
    centroids = l2normalize(initClusts);
    maxRestarts = 0;
else
    error('emonmf: initClusts is not valid.');
end

if any(M<0)
    error('M must be nonnegative')
end

MnormSq = sum(sum(M.*M));
pdf = full(sum(M.^2,2)/MnormSq);

keepRestarting = 1;
nRestarts = 0;
hasRestarted = 0;
lastPrintedString = sprintf('\nEM-onmf: 1/%d',MaxIters);
fprintf(lastPrintedString);
while nRestarts <= maxRestarts && keepRestarting
    if hasRestarted
        fprintf('\n\nRestart n. %d\n',nRestarts);
        lastPrintedString = '';
    end
    if ~areInitCentroidsGiven
        centroids = getInitCentroids(M,numClusts,pdf,initMethod);
    end
    lastLabel = repmat(-1,numItems,1);
    hasLabelChanged = 1;
    iter = 1;
    while hasLabelChanged && iter <= MaxIters
        lastPrintedString = printIters(lastPrintedString,iter,MaxIters);
        labels = updateLabels(M,centroids);
        centroids = updateCentroids(M,labels,numClusts);
        [labels,centroids] = handleEmptyClusters(M,labels,centroids);
        hasLabelChanged = any(lastLabel-labels);   
        lastLabel = labels;
        iter = iter + 1;
    end
    iter = iter - 1;% to compensate for last added (but not executed) iter
    if iter < MaxIters
        keepRestarting = false; % iterations were good: convergence in less than MaxIters
    else
        nRestarts = nRestarts + 1;
        hasRestarted = 1;        
    end    
end
if nargout > 2
    relError = 0;
    for k = 1:numClusts
        idxi = labels == k;
        Mi = (M(idxi,:));
        ui = centroids(k,:);
        vi = Mi*ui';
        partialErr = sum(sum( Mi.*Mi )) - 2*ui*Mi'*vi + (vi'*vi)*(ui*ui');
        relError = relError + partialErr;
    end    
    relError = relError / MnormSq;
end
if nargout > 4
    U_binarized = binarizeInd(labels,numClusts);
end
fprintf('\n');

%==========================================================================

function labels = updateLabels(M,centroids)

[vals,labels] = max(M*centroids',[],2);

%==========================================================================

function centroids = updateCentroids(M,labels,numClusts)

numFeatures = size(M,2);
centroids = zeros(numClusts,numFeatures);
for k = 1:numClusts
    idxi = labels == k;
    if any(idxi)
        Mi = (M(idxi,:));
        [ui,si,vi] = svds(Mi',1);
        centroids(k,:) = abs(ui);
    else
        centroids(k,:) = NaN(numFeatures,1);
    end
end

%==========================================================================

function [labels,centroids] = handleEmptyClusters(M,labels,centroids)

numClusts = size(centroids,1);
numItems = size(M,1);
missingLabels = setdiff(1:numClusts,labels); % check for empty clusters
if any(missingLabels)
    singlIdx = randsample(numItems,length(missingLabels));
    labels(singlIdx) = missingLabels;
    centroids = updateCentroids(M,labels,numClusts);
end

%==========================================================================

function centroids = getInitCentroids(M,numClusts,pdf,initMethod)

switch initMethod 
    case 1
        centroids = l2normalize(rand(numClusts,numFeatures)); % random values
    otherwise % DEFAULT init: subset of data points    
        centroids = getRandomCentroids(M,numClusts,pdf);
end

%==========================================================================

function centroids = getRandomCentroids(M,numClusts,pdf)

% cidx = find(mnrnd(numClusts,pdf));
numItems=size(M,1); cidx = randsample(numItems,numClusts);
centroids = l2normalize(M(cidx,:)); 

%==========================================================================

function normRowMat = l2normalize(rowMat)

rowMatNorm = sum(rowMat.^2,2).^(-1/2);
normRowMat = bsxfun(@times,rowMat,rowMatNorm);
normRowMat(isnan(normRowMat)) = 0;

%==========================================================================

function U = binarizeInd(labels,numClusts)

numItems = length(labels);
U = full(sparse(1:numItems,labels,ones(numItems,1),numItems,numClusts));

%==========================================================================

function stringIters = printIters(lastPrintedString,iter,MaxIters)

if mod(iter,1)==0
    asciiDelete = double('\b');            
    stringBS = char(repmat(asciiDelete,1,length(lastPrintedString)));
    fprintf(stringBS);
    stringIters = sprintf('EM-onmf: %d/%d',iter,MaxIters);
    fprintf(stringIters);    
end



