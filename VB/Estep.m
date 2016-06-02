%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function model = Estep(fea, model)
% estimate responsibilities by derivation of posterior distribution w.r.t model parameter
%
% fea:	d*n matrix
% 	d:	number of dimension 
% 	n:	number of data
% model:	parameters of GMMs
% Author Ziyi Guo(zig312@lehigh.edu)

alpha = model.alpha;
beta = model.beta;
m = model.m;
v = model.v;
W = model.W;

[d,n] = size(fea);
k = size(m,2);

logdet = zeros(1,k);
ELogQuadra = zeros(n,k);

for i=1:k
	U = chol(W(:,:,i));
	logdet(i) = -2*sum(log(diag(U))); % 2 means |AB| = |A|*|B| negative means the inverse of W matrix
	sqrtR = (U'\bsxfun(@minus,fea,m(:,i)));
	ELogQuadra(:,i) = d/beta(i)+v(i)*dot(sqrtR,sqrtR,1); % equation(10.64) in Bishop's PRML
end 

ELogPrecision = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logdet; % equation(10.65) in Bishop's PRML
ELogPi = psi(0,alpha)-psi(0,sum(alpha)); % equation(10.66) in Bishop's PRML

LogPnk = (bsxfun(@minus,ELogQuadra,2*ELogPi+ELogPrecision-d*log(2*pi)))/(-2); % equation(10.46) in Bishop's PRML
LogRnk = bsxfun(@minus,LogPnk,logsumexp(LogPnk,2));
responsibilities = exp(LogRnk); % equation(10.49) in Bishop's PRML

model.LogRnk = LogRnk;
model.responsibilities = responsibilities;
