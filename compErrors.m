function [errors] = compErrors(confmx)
% confmx - confusion matrix of the classifier
% errors - vector containing proper classifications, errors and reject decision coefficients (in %)
	total = sum(sum(confmx));
	errors(1) = trace(confmx) / total;
	errors(2) = (total - trace(confmx) - sum(confmx(:,end))) / total;
	errors(3) = sum(confmx(:,end)) / total;
