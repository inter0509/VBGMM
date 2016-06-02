%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function model = Mstep(fea, model, prior)
% estimate model parameters by derivation of posterior distribution w.r.t latent variables

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
responsibilities = model.responsibilities;

Nk = sum(responsibilities,1); % equation(10.51) in Bishop's PRML
xsum = fea*responsibilities;
xbar = bsxfun(@times,xsum,1./Nk);

alpha = alpha0 + Nk; % equation(10.58) in Bishop's PRML
beta = beta0 + Nk; % equation(10.60) in Bishop's PRML
m = bsxfun(@times,bsxfun(@plus,beta0*m0,xsum),1./beta); % equation(10.61) in Bishop's PRML
v = v0 + Nk; % equation(10.63) in Bishop's PRML

[d,n] = size(fea);
k = size(m,2);
W = zeros(d,d,k);

sqrtR = sqrt(responsibilities);
xbarm0 = bsxfun(@minus,xbar,m0);
term = beta0*Nk./(beta0+Nk);
for i =1:k
	SkNk = bsxfun(@times,bsxfun(@minus,fea,xbar(:,i)),sqrtR(:,i)'); % equation(10.53) in Bishop's PRML
	W(:,:,i) = W0+SkNk*SkNk'+(term(i))*(xbarm0(:,i)*xbarm0(:,i)'); % equation(10.62) in Bishop's PRML
end

model.alpha = alpha;
model.beta = beta;
model.m = m;
model.v = v;
model.W = W;

