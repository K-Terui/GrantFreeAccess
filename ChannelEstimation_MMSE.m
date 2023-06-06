function NMSE = ChannelEstimation_MMSE(A, y, h, xhat, SetEst, nvar)
% Channel estimation method with MMSE for grant free access
% After the channel estimation, calculate the NMSE

% initialize
M = size(A, 1);
Aest = A(:, SetEst);

% MMSE
xmmse = Aest' * inv(Aest * Aest' + nvar * eye(M)) * y;

% sparse vector reconstruction
xhat(SetEst) = xmmse;

% NMSE calculation
NMSE = norm(h - xhat, 'fro')^2 / norm(h, 'fro')^2;
end