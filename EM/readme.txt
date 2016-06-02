1. Run EM algorithm
	load faithful;
	[label,model,BIC] = GmmFit(x,2);
	
2. Plot BIC vs. the number of iteration
	plot(BIC,'b--o');
	
3. Plot original data and clustered data
	DataPlot(x,label);
	
4. Plot average BIC vs. the choice of clusters
	BIC = BicPlot();