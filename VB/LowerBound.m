%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function LBound = LowerBound(fea,model,prior)
% estimate variational lower bound for model selection

% fea:	d*n matrix
% 	d:	number of dimension 
% 	n:	number of data
% model:	parameters of GMMs
% prior:	prior distributions of hyper-parameters
% Author Ziyi Guo(zig312@lehigh.edu)

alpha0 = prior.alpha;
beta0 = prior.beta;
m0 = prior.m;
v0 = prior.v;
W0 = prior.W;

alpha = model.alpha;
beta = model.beta;
m = model.m;
v = model.v;
W = model.W;
responsibilities = model.responsibilities;
LogRnk = model.LogRnk;

[d,n] = size(fea);
k = size(m,2);

Nk = sum(responsibilities,1); % equation(10.51) in Bishop's PRML
ELogPi = psi(0,alpha)-psi(0,sum(alpha)); % equation(10.66) in Bishop's PRML

E_PZ = dot(Nk,ELogPi); % equation(10.72) in Bishop's PRML
E_QZ = dot(responsibilities(:),LogRnk(:)); % equation(10.75) in Bishop's PRML

E_PPi = gammaln(k*alpha0)-k*gammaln(alpha0)+(alpha0-1)*sum(ELogPi); % equation(10.73) in Bishop's PRML
E_QPi = gammaln(sum(alpha))-sum(gammaln(alpha))+dot(alpha-1,ELogPi); % equation(10.76) in Bishop's PRML

%fprintf('%f\n',E_PZ+E_PPi-E_QZ-E_QPi);

logW = zeros(1,k);
TrSW = zeros(1,k);
TrW0Wk = zeros(1,k);
Quadra1 = zeros(1,k);
Quadra2 = zeros(1,k);

U0 = chol(W0);
sqrtR = sqrt(responsibilities);
xbar = bsxfun(@times,fea*responsibilities,1./Nk);

for i=1:k
	U = chol(W(:,:,i));
	logW(i) = -2*sum(log(diag(U)));
	
	sqrtSk = bsxfun(@times,bsxfun(@minus,fea,xbar(:,i)),sqrtR(:,i)');
	Sk = (sqrtSk*sqrtSk'/Nk(i)); % equation(10.53) in Bishop's PRML
	
	Sk = chol(Sk);
	
	sqrtSW = Sk/U;
	TrSW(i) = dot(sqrtSW(:),sqrtSW(:));
	sqrtW0W = U0/U;
	TrW0Wk(i) = dot(sqrtW0W(:),sqrtW0W(:));
	
	sqrtQua1 = U'\(xbar(:,i)-m(:,i));
	Quadra1(i) = dot(sqrtQua1,sqrtQua1);
	sqrtQua2 = U'\(m(:,i)-m0);
	Quadra2(i) = dot(sqrtQua2,sqrtQua2);
end

ELogPrecision = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logW; % equation(10.65) in Bishop's PRML
logB0 = v0*sum(log(diag(U0)))-0.5*v0*d*log(2)-logmvgamma(0.5*v0,d);
E_PMu = 0.5*sum(d*log(beta0/(2*pi))+ELogPrecision-d*beta0./beta-beta0*(v.*Quadra2))+k*logB0+0.5*(v0-d-1)*sum(ELogPrecision)-0.5*dot(v,TrW0Wk); % equation(10.74) in Bishop's PRML

logB = -0.5*(v.*(logW+d*log(2)))-logmvgamma(0.5*v,d);
sumH = -0.5*sum((v-d-1).*ELogPrecision)+0.5*sum(v*d)-sum(logB);
E_QMu = 0.5*sum(ELogPrecision+d*log(beta/(2*pi)))-0.5*d*k-sumH; % equation(10.77) in Bishop's PRML

E_PX = 0.5*dot(Nk,ELogPrecision-d./beta-v.*TrSW-v.*Quadra1-d*log(2*pi)); % equation(10.71) in Bishop's PRML

% add all components
LBound = E_PX+E_PZ+E_PPi+E_PMu-E_QZ-E_QPi-E_QMu;
LBound = LBound + sum(log(1:k));



