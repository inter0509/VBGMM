%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function DataPlot(x, label)
% plot data points with clustering labels
%
% x:	input data
% label:	clustering results
%
% Author Ziyi Guo(zig312@lehigh.edu)

figure;
plot(x(1,:),x(2,:),'*');

n = size(x,2);
labelSet = unique(label);
colors = jet(length(labelSet));

figure;
for i=1:n
	color = labelSet(label(i));
	plot(x(1,i),x(2,i),'*','Color',colors(color,:));
	hold on;
end