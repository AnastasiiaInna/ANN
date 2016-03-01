function O = sigmoid(I)
  % Computes the sigmoid of the given net input I.

  O = 1.0 ./ (1.0 + exp(-I));
end