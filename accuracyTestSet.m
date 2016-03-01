function [MSE Accuracy Error Rejected] = accuracyTestSet(tvec, tlab, wInputHidden, wHiddenOutput)
  
  incorrectSamples = 0;
  rejectedSamples = 0;
  MSE = 0;
  
  treshold = 0.6; %0.9;  
  trainSize = rows(tvec);
    
  nInputs = columns(tvec);  
  nHidden = columns(wInputHidden);
  nOutputs = columns(wHiddenOutput);

  for i = 1:trainSize
    
    inputNeurons = (tvec(i, :))';   
    desiredValue = zeros(size(unique(tlab), 1), 1);
    desiredValue(tlab(i)) = 1;
    
    [hiddenNeurons outputNeurons] = feedForward(inputNeurons, wInputHidden, wHiddenOutput); 
   
    MSE += sum((outputNeurons - desiredValue) .^ 2); 
    
    [mv mi] = max(outputNeurons); 
    secondMax = max(outputNeurons(outputNeurons != mv));
    if ((mv < treshold) || (mv < 1.2 * secondMax))
     % if (mv < treshold) 
      rejectedSamples += 1;
     elseif (mi != tlab(i)) 
    %if (mi != tlab(i)) 
      incorrectSamples += 1;
    end
  end
  
  MSE = MSE / (nOutputs * trainSize);  
  Rejected = rejectedSamples / trainSize * 100;
  Error = incorrectSamples / trainSize * 100;
  Accuracy = 100 - Error - Rejected;
end