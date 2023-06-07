function [Xhat, SetEst] = OMP_nzknown(Y, A, K)
% OMP algorithm (Case: num. non-zero elements is known at the receiver)
% min. ||y - A\hat{x}||_0, s.t. y = Ax + n, ||x||_0 = K 
% input
% Y      : received signal (matrix)
% A      : mesurement matrix
% K      : num. non-zero elements 
% output
% xhat   : reconstructed sparse vectors
% SetEst : indeces set of reconstructed sparse vector


% initialize
J      = size(Y, 2);
Xhat   = zeros(size(A, 2), J); %reconstructed vectors (sparse) of all antennas
SetEst = int16.empty;          %index set of estimated active UEs of all antennas

% main loop
for j = 1 : J
    % initialize
    xhat_j   = zeros(size(A, 2), 1); %reconstructed vector (sparse) of the j-th antenna  
    SetEst_j = int16.empty;          %index set of estimated active UEs of the j-th antenna
    A_hat  = double.empty;           %selected support
    y = Y(:, j);                     %recived signal

    for k = 1 : K
        % residual update
        r = y - A * xhat_j;
    
        % select the maximum correlated index
        [~, p] = max(abs(A' * r));
        
        % add index to the index set
        SetEst_j(k) = p;
        
    
        % derive the least square solution
        A_hat(:, k) = A(:, SetEst_j(k));
        xtilde  = pinv(A_hat) * y;
    
        % sparse reconstruction
        xhat_j(SetEst_j) = xtilde;
    
    end
    % data storing
    SetEst(:, j) = SetEst_j;
    Xhat  (:, j) = xhat_j;
    
end

end