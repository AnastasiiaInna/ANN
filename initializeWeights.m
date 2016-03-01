function W = initializeWeights(nInputs, nOutputs)
% Initializes the weights of a layer with "nInputs" inputs and "nOutputs" outputs

% W is a matrix of size(1 + nInputs, nOutputs) as the first row of W handles the "bias" terms
W = zeros(1 + nInputs, nOutputs); 

epsilon = 1; % for ranging; could be from -1 to 1, or -0.5 to 0.5
W = rand(1 + nInputs, nOutputs) * 2 * epsilon - epsilon; 

beta = 0.7 * power(nOutputs, 1 / nInputs);
W = W * beta ./ sqrt(sum(W .^ 2, 2));
end