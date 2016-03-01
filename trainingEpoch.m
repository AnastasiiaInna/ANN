function [MSE Accuracy Error Rejected wInputHidden wHiddenOutput] = trainingEpoch(tvec, tlab, wInputHidden, wHiddenOutput, learningRate, momentum)
  
  incorrectSamples = 0;
  rejectedSamples = 0;
  MSE = 0;
  
  treshold = 0.7;%0.9; % 0.7; 
  trainSize = rows(tvec);
    
  nInputs = columns(tvec);  
  nHidden = columns(wInputHidden);
  nOutputs = columns(wHiddenOutput);
  % wInputHidden = initializeWeights(nInputs, nHidden);   % first row of W handles the "bias" terms
  % wHiddenOutput = initializeWeights(nHidden, nOutputs); % first row of W handles the "bias" terms

  for i = 1:trainSize
    
    inputNeurons = (tvec(i, :))';   
    desiredValue = zeros(size(unique(tlab), 1), 1);
    desiredValue(tlab(i)) = 1;
    
    [hiddenNeurons outputNeurons] = feedForward(inputNeurons, wInputHidden, wHiddenOutput); 
 
    deltaBiasInputHidden = zeros(nHidden, 1);
    deltaBiasHiddenOutput = zeros(nOutputs, 1);
    deltaInputHidden = zeros(nInputs, nHidden);
    deltaHiddenOutput = zeros(nHidden, nOutputs);
    
    [deltaHiddenOutput deltaInputHidden deltaBiasHiddenOutput deltaBiasInputHidden] = deltaWeightsBiases(inputNeurons, hiddenNeurons, outputNeurons, wHiddenOutput, deltaHiddenOutput, deltaInputHidden, desiredValue, learningRate, momentum);
    
    wHiddenOutput(2:end, :) += deltaHiddenOutput; % Update weights
    wHiddenOutput(1, :) += deltaBiasHiddenOutput; % Update biases
    wInputHidden(2:end, :) += deltaInputHidden; % Update weights
    wInputHidden(1, :) += deltaBiasInputHidden; % Update biases
    
    MSE += sum((outputNeurons - desiredValue) .^ 2); 
    
    [mv mi] = max(outputNeurons); 
     secondMax = max(outputNeurons(outputNeurons != mv));
     if ((mv < treshold) || (mv < 1.2 * secondMax))
     %if (mv < treshold)
      rejectedSamples += 1;
     elseif (mi != tlab(i)) 
    %if (mi != tlab(i)) 
      incorrectSamples += 1;
    end
  end
  
  MSE = MSE / (nOutputs * trainSize);  
  Rejected = rejectedSamples / trainSize ;
  Error = incorrectSamples / trainSize;
  Accuracy = 1 - Error - Rejected;
end