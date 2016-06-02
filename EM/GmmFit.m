
%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function [label,model,BIC] = GmmFit(fea,k)
% Fit GMM parameters iteratively with E steps and M steps
%
% fea:	d*n matrix
% 	d:	number of dimension 
% 	n:	number of data
% k:	number of clusters
%
% Author Ziyi Guo(zig312@lehigh.edu)

responsibility = EmInitial(fea,k);

% parameters initializations
tol = 1e-10;
maxiter = 1000;
BIC = -inf(1,maxiter);
converged = false;
iter = 1;

while ~converged && iter < maxiter
	iter = iter+1;
	model = Mstep(fea,responsibility);
	[responsibility, BIC(iter)] = Estep(fea,model);
	[~,label] = max(responsibility,[],2);
	
	converged = (BIC(iter)-BIC(iter-1) < tol*abs(BIC(iter)));
end

BIC = BIC(2:iter);