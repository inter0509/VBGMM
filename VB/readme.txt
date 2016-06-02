1. Run VB algorithm
	load faithful;
	[label, model, L, K] = VBfit(x,6);
		  label:the clustering result
	      model:the estimated parameters
		  L: the lower bound at each iteration
		  K: the optimal number of component
	
2. Plot lower bound  vs. the number of iteration
	plot(L,'b--o');