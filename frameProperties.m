function [mc, wb] = frameProperties(A)
% calculate the mutual coherence and welch bound of arbitrary frame A
% Input
% A   :  The arbitrary frame, num. rows is M, num. columun N
% Output
% mc  :  Mutual coherence
% wb  :  Welch bound

% initialization
[M, N] = size(A);

if (M>=N)
    msg = 'Error! The number of columns must be greater than the number of rows.';
    error(msg);
end

% mutual coherence calculation
mc = max(abs(A'*A.*(1-eye(N))),[],'all');

% welch bound calculation
wb = sqrt((N-M)./(M.*(N-1)));

end