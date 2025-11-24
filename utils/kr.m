function C = kr(A, B)
%Khatri Rao Product - Column-wise kronecker Product
%==================================================
% Input:
%   A: m x n
%   B: l x n
% Out:
%   C: (lm) x n
%==================================================
% Implemented by Qilun Luo, 2023

A = permute(A, [3, 1, 2]);
B = permute(B, [1, 3, 2]);
C = pagemtimes(B, A);
C = reshape(C, [], size(C, 3));
