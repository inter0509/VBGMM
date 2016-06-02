%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function BIC = BicPlot();
%  plot for a range choice of k for model selection
%
%  Author Ziyi Guo(zig312@lehigh.edu)

load faithful;

K = [1:10]; % Number of Gaussian components you want to test
R = 20; % NUmber of repetition to run

BIC = zeros(R,length(K));

figure; 

for rep = 1:R
    cnt_k = 1;
    for k = K
		[label,model,B] = GmmFit(x,k);
        BIC(rep,cnt_k)= B(end-1);
        cnt_k = cnt_k + 1;
    end
end

meanBic = mean(BIC,1);
plot(meanBic,'--o','MarkerSize',10,'LineWidth',1.5);