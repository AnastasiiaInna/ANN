function cfmx = confMx(trueLab, decLab)
% Compute confusion matrix
% trueLab - state of anture labels for classified set
% decLab - labels produced by classifier

  labels = unique(trueLab);
  cfmx = zeros(rows(labels), rows(labels) + 2);
  for i = 1:rows(trueLab)
    cfmx(trueLab(i), decLab(i)) += 1;
  end