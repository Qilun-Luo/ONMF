function X = Prox_Pos_L1(T, tau)
% Proximal operator for positive l1 regularization
% \min_{X} 1/2 \|X- T\|_F^2 + \tau \|(X)_+\|_1 
% =========================================================================
%  Input:
%   T: the given matrix
%   tau: regularization parameter
%  Output:
%   X: solution
% =========================================================================
% Implemented by Qilun Luo, Nov. 13, 2023

X = max(T-tau, (T<=tau).*min(T, 0));

end