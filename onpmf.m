% This algorithm tries to solve
%
% min_{U,V}   ||M-UV||_F^2
%       such that U >= 0, V >= 0 and U'U = I.
%
% It alternates the minimization w.r.t. V and U in two steps: 1) V
% minimization is done by solving a nonnegative least-squares subproblem.
% 2) U minimization is performed by using a penalty method, i.e., by
% solving min_{U,V}   ||M-UV||_F^2 + rho*||max(-U,0)||_F^2
%       such that V >= 0 and U'U = I
% for increasing values of rho. 
%
%
%
% Written by:
% Filippo Pompili (filippopompili.09@gmail.com), and,
% Nicolas Gillis (https://sites.google.com/site/nicolasgillis/).
%
% Reference paper: 
% F. Pompili, N. Gillis, P.-A. Absil, F. Glineur,
% "Two algorithms for orthogonal nonnegative matrix 
% factorization with application to clustering".
% Neurocomputing, Vol. 141, pp. 15-25, 2014.
%
% This code comes with no guarantee or warranty of any kind.
%


function [U,V,error,iter] = onpmf(M,r,niters,U,V,timelimit)

% User Settings

C = 1.01;
alpha0 = 1e2;
rho = 1e-2;

minOpts.maxMinIters = 1;
lsOpts.stepBeta = 1.1;
lsOpts.stepLowerThreshold = 1e-15;

relNegativityThreshold = 1e-3;
rhoUpperThreshold = 1e100;
maxStepLowCount = 5;
stepLowerThresholdCount = 1e-14;
maxTimeLimit = 7200; % seconds








%==========================================================================
M(M<0) = 0;

if nargin < 4
    U = svdInit(M,r);
else
    U = closestOrthogonalMatrix(U);
end
if nargin < 5
    V = updateV(M,U);    
end
if nargin < 6
    timelimit = maxTimeLimit;
end

MnormSq = sum(sum(M.*M));
Mnorm = sqrt(MnormSq);
initialtime = cputime;
elapsedTime = cputime-initialtime;
stepUHist = zeros(niters,1);
errfitHist = zeros(niters,1);
errnegHist = zeros(niters,1);
N = zeros(size(U));
stepLowStopflag = 0;
stepLowCount = 0;
stepU = 1;
iter = 1;

fprintf('onpmf ls:init\tplty:%1.2e\tmisft:%1.2e\tneg:%1.2e\n',rho,computeFitError(M,U,V,MnormSq),computeNegValsRelNorm(U));

while iter <= niters &&  (iter < 2 || errnegHist(iter-1) > relNegativityThreshold) && stepLowStopflag == 0 && elapsedTime <= timelimit
   
    V = updateV(M,U);
    rho = updateRho(C,rho,rhoUpperThreshold);
    [U,stepU] = updateU(M,U,V,N,rho,stepU,MnormSq,minOpts,lsOpts);
    alpha = updateAlpha(iter,alpha0); N = updateN_alpha(alpha,U,N);
    
    % step stopping condition        
    stepLowStopflag = checkStepConvergence(stepU,stepLowCount,stepLowerThresholdCount,maxStepLowCount);

    stepUHist(iter) = stepU;
    errfitHist(iter) = sqrt(computeFitError(M,U,V,MnormSq))/Mnorm;
    errnegHist(iter) = computeNegValsRelNorm(U);
    elapsedTime = cputime-initialtime;
    
    if mod(iter,100)==0
        fprintf('onpmf ls:%4d/%4d\tplty:%1.2e\tmisft:%1.2e\tneg:%1.2e\n',iter,niters,rho,errfitHist(iter),errnegHist(iter));     
    end
    
    iter = iter + 1;    
end

iter = iter - 1;
% hf = figure; hp = semilogy(stepUHist); grid on; axis square; set(hf,'Color','white'); set(hp,'Color','black','LineWidth',2.0)
% hf = figure; hp = semilogy(errnegHist); grid on; axis square; set(hf,'Color','white'); set(hp,'Color','black','LineWidth',2.0)
% hf = figure; hp = semilogy(errfitHist); grid on; axis square; set(hf,'Color','white'); set(hp,'Color','black','LineWidth',2.0)
error = errfitHist(iter-1);

fprintf('\niterations : %d\n',iter-1);
fprintf('\n');
fprintf('penalty parameter: %2.0g\n',rho);
fprintf('relative error: %2.3f\n',errfitHist(iter-1));
fprintf('relative negtv. constr. residual: %2.4g\n',errnegHist(iter-1));
fprintf('\n');


%==========================================================================
function stepLowStopflag = checkStepConvergence(step,stepLowCount,stepLowerThresholdCount,maxStepLowCount)

stepLowStopflag = 0;
if step <= stepLowerThresholdCount
    stepLowCount = stepLowCount + 1;
end            
if stepLowCount >= maxStepLowCount
    fprintf('\n Step has converged. \n');
    stepLowStopflag = 1;
end

%==========================================================================
function U = closestOrthogonalMatrix(A)

% computes U s.t. A = U*P, U'*U = I
P = sqrtm(A'*A);
U = A/P;

%==========================================================================
function fitErr = computeFitError(M,U,V,MnormSq)

fitErr = 1/2*MnormSq - sum(sum( (M*V').*U )) + 1/2*sum(sum( (V*V').*(U'*U) ));

%==========================================================================
function lagrValue = computeLagrangianValue(M,U,V,rho,N,MnormSq)

fitErr = computeFitError(M,U,V,MnormSq);
constrError = sum(sum(N.*(-U))) + 1/2*rho*norm(U(U<0),'fro')^2;
lagrValue = fitErr + constrError;

%==========================================================================
function relNorm = computeNegValsRelNorm(U)

relNorm = norm(U(U<0),'fro')/norm(U,'fro');

%==========================================================================
function rho = updateRho(C,rho,rhoUpperThreshold)

if rho < rhoUpperThreshold
    rho = C*rho;
end

%==========================================================================
function N = updateN_alpha(alpha,U,N)

N = max(0,N-alpha*U);

%==========================================================================
function alpha = updateAlpha(iter,alpha0)

alpha = alpha0/iter;

%==========================================================================
function [U,step,lagrValue] = updateU(M,U,V,N,rho,step,MnormSq,minOpts,lsOpts)

agrad = @(U)(M*V'- U*(V*V')) + (N + rho*max(-U,0));
stepMove = @(U,step)stepMoveU(U,step,agrad);
computeLagrangianValueForX = @(U)computeLagrangianValue(M,U,V,rho,N,MnormSq);

[U,step,lagrValue] = minimizeFun(U,step,stepMove,computeLagrangianValueForX,minOpts,lsOpts);

%==========================================================================
function U_new = stepMoveU(U,step,agrad)

U_euc = U + step*agrad(U);
U_new = closestOrthogonalMatrix(U_euc);

%==========================================================================
function [x,step,lagrValue] = minimizeFun(x,step,stepMove,computeLagrangianValueForX,minOpts,lsOpts)

tolFun = 1e-4;

diffFun = +Inf; % difference in subsequent function values
iter = 1;
while diffFun > tolFun && iter <= minOpts.maxMinIters    
    [x,step,lagrValue,lagrVal0] = lineSearch(x,step,stepMove,computeLagrangianValueForX,lsOpts);    
    diffFun = abs(lagrVal0 - lagrValue);
    iter = iter + 1;
end % outer iter

%==========================================================================
function [x,step,lagrValue,startLagrangValue] = lineSearch(x,step,stepMove,computeLagrangianValueForX,lsOpts)

startLagrangValue = computeLagrangianValueForX(x);
lastLagrangVal = startLagrangValue;
isStepAccepted = 0;
j = 1;

while ~isStepAccepted && step > lsOpts.stepLowerThreshold
    x_new = stepMove(x,step);
    lagrVal_candidate = computeLagrangianValueForX(x_new);
    hasImproved = lagrVal_candidate < startLagrangValue && lagrVal_candidate < lastLagrangVal;
    if j == 1
       keepIncreasing = hasImproved;           
       last_x = x_new;
    end
    if keepIncreasing
        if hasImproved
            last_x = x_new;
            lastLagrangVal = lagrVal_candidate;
            step = lsOpts.stepBeta*step;
        else                
            step = step/lsOpts.stepBeta;
            x = last_x;
            isStepAccepted = 1;
        end                
    else
        if hasImproved
            lastLagrangVal = lagrVal_candidate;
            x = x_new;
            isStepAccepted = 1;
        else
            step = step/lsOpts.stepBeta;
        end
    end
    j = j + 1;
end % end line search
lagrValue = lastLagrangVal;

%==========================================================================
function V = updateV(M,U)

V = nnlsm_blockpivot(U,M);

%==========================================================================
function U = svdInit(M,r)

n = size(M,2);
if n < r
    [U,Sigma,Vt] = svd(M); U = U(:,1:r);           
else
    [U,Sigma,Vt] = svds(M,r);
end
% sign flip ambiguity check
for j=1:r
    negidx = U(:,j)<0;
    isNegNormGreater = norm(U(negidx,j),'fro') > norm(U(~negidx,j),'fro');
    if isNegNormGreater
        U(:,j) = -U(:,j);
    end  
end


