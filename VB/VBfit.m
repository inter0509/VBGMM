%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function [label, model, L, K] = VBfit(fea, initK)
% Fits VBGMM parameters
%
% fea:	d*n matrix
% 	d:	number of dimension 
% 	n:	number of data
% initK:	number of clusters
%
% Author Ziyi Guo(zig312@lehigh.edu)
	
fprintf('Variational Bayesian Gaussian mixture: running ... \n');
[d,n] = size(fea);

if nargin < 3
    prior.alpha = 0.01;
    prior.beta = 1;
    prior.m = mean(fea,2);
    prior.v = d+2;
    prior.W = eye(d);
end

model.responsibilities = VBInitial(fea,initK);

% data initializations
tol = 1e-20;
maxiter = 200;
L = -inf(1,maxiter);
converged = false;
iter = 1;

% VB EM iterations 
while  ~converged && iter < maxiter
    iter = iter+1;
    model = Mstep(fea, model, prior);
    model = Estep(fea, model);
    L(iter) = LowerBound(fea,model,prior);
    converged = abs(L(iter)-L(iter-1)) < tol*abs(L(iter));
end

L = L(2:iter);
label = zeros(1,n);
[~,label(:)] = max(model.responsibilities,[],2);
fprintf('Converged in %d steps.\n',iter);

% computes the effective number of clusters
K = initK - sum(sum((model.responsibilities-0))<0.001);