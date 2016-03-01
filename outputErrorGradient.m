function outputError = outputErrorGradient(outputNeurons, desiredValue)
% Fucntion for computing the error of each unit in the output layer;
% outputNeurons are the actual outputs;
% trueLabels are true outputs, based on the known class label of the given training samples;


  % outputError = zeros(length(outputNeurons), 1);
  outputError = outputNeurons .* (1 - outputNeurons) .* (desiredValue - outputNeurons);