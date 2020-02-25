function ix = win(w,x)
%% Identify winning neuron from input and vectors of weights.

	[~,neurons] = size(w);
	dist = zeros(1,neurons);
        
	% Repeat Weight Vector of Input to Match With w Matrix Dimensions.
	xn_rep = repmat(x',1,neurons);
        
	% Search minor euclidean distance.
	dist(1,1:neurons) = ( sum( (w - xn_rep ).^2 ) ).^0.5;
	[~,winning] = min(dist);

	ix = winning;
end
