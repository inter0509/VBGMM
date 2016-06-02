%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function [responsibility,BIC] = Estep(fea, model)
% Given parameters, estimate new value of responsibilities via posterior computation
%
% fea:	d*n matrix
% 	d:	number of dimension 
% 	n:	number of data
% model:	parameters of GMM
%
% Author Ziyi Guo(zig312@lehigh.edu)
mu = model.mu;
Sigma = model.Sigma;
weight = model.weight;

[d,n] = size(fea);
k = size(mu,2);
logTerm1 = zeros(n,k);

for i = 1:k
    logTerm1(:,i) = loggausspdf(fea,mu(:,i),Sigma(:,:,i)); %log likelihood of each MVN distribution
end
logTerm1 = bsxfun(@plus,logTerm1,log(weight));
logTerm2 = logsumexp(logTerm1,2); %equation(9.28) in Bishop's PRML

M = (d*(d+3)/2+1)*k; %number of parameters in total
BIC = sum(logTerm2)-0.5*(M)*log(n); % Bayesian information criterion (BIC) computation 

logR = bsxfun(@minus,logTerm1,logTerm2); %equation(9.39) in Bishop's PRML
responsibility = exp(logR);



%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function y = loggausspdf(X, mu, Sigma)
% A helper function that computes log likelihood of one gaussian distribution
d = size(X,1);
X = bsxfun(@minus,X,mu);
[U,p]= chol(Sigma);
%if p ~= 0
%    error('ERROR: Sigma is not PD.');
%end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;



%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function s = logsumexp(x, dim)
% A helper function that computes log(sum(exp(x),dim))
if nargin == 1, 
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end
