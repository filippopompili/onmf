% Written by Filippo Pompili
% (filippopompili.09@gmail.com).
%

load hubble;

maxOnpmfIters = 3000;
maxEmIters = 100;
emThreshold = 1e-9;





%==========================================================================
% Dataset related config
M(M<0) = 0;
numClusters = 8;
height = 128;
width = height;


% ONPMF
[Uonpmf,V,relError,actualIters] = onpmf(M,numClusters,maxOnpmfIters);
showMatrixColumns(Uonpmf,numClusters,height,width);


% EM-ONMF
% Em preprocessing to remove background low values. 
Mnorm = sum(M.^2,2).^(1/2);
removeidx = find(Mnorm <= emThreshold);
keepidx = setdiff((1:size(M,1))',removeidx);
if isempty(keepidx)
    error('ImageDatasetLoader:badEmThreshold','Too high EmThreshold. No values remaining after filtering.');
end;
Mreduced = M(keepidx,:);
% Actual Clustering
[clusters,V,relError,actualIters] = emonmf(Mreduced,numClusters,maxEmIters);
% Inflating solution..
Uem = zeros(size(M,1),numClusters);
for k = 1:numClusters
   clusterIdx = clusters == k;
   Uem(keepidx(clusterIdx),k) = k;
end
showMatrixColumns(Uem,numClusters,height,width);


