%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function responsibility = EmInitial(fea,k)
% Initialize GMM parameters with K-means clustering
%
% fea:	d*n matrix
% 	d:	number of dimension 
% 	n:	number of data
% k:	number of clusters
%
% Author Ziyi Guo(zig312@lehigh.edu)

[d,n] = size(fea);

idx = randsample(n,k);
m = fea(:,idx);
[~,label] = max(bsxfun(@minus,m'*fea,dot(m,m,1)'/2),[],1);
[u,~,label] = unique(label);

% K-means iterations 
while k ~= length(u)
    idx = randsample(n,k);
    m = fea(:,idx);
	% update cluster labels
    [~,label] = max(bsxfun(@minus,m'*fea,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
end

responsibility = full(sparse(1:n,label,1,n,k,n));