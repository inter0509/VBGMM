%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function model = Mstep(fea, responsibility)
% Given responsibilities, estimate new value of parameters via maximization of log likelihood function
%
% fea:	d*n matrix
% responsibility: n*k
% 	d:	number of dimension 
% 	n:	number of data
% k:	number of clusters
%
% Author Ziyi Guo(zig312@lehigh.edu)

[d,n] = size(fea);
k = size(responsibility,2);

Nk = sum(responsibility,1); %equation(9.27) in Bishop's PRML
weight = Nk/n; %equation(9.26) in Bishop's PRML
mu = bsxfun(@times, fea*responsibility, 1./Nk); %equation(9.24) in Bishop's PRML

Sigma = zeros(d,d,k);
sqrtR = sqrt(responsibility);
for i = 1:k
    Xo = bsxfun(@minus,fea,mu(:,i));
    Xo = bsxfun(@times,Xo,sqrtR(:,i)');
    Sigma(:,:,i) = Xo*Xo'/Nk(i); %equation(9.25) in Bishop's PRML
    Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-6); % add a prior for numerical stability
end

model.mu = mu;
model.Sigma = Sigma;
model.weight = weight;