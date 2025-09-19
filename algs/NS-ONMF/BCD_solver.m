function [X, Z] = BCD_solver(A, r, X, Z, rho, L, eps, Gamma)
%BCD_solver to solve the subproblem for W = (X, Z) by the 
% block coordinate descent (BCD) algorithm.
% =========================================================================
%  Input:
%   A: m x n matrix 
%   r: rank
%   X: previous n x n matrix
%   Z: n x (n-r) matrix
%   rho: penalty 
%   L: Lagrangian multiplier 
%   eps: tolerance
%   Gamma: bound of Lagrangian function
%  Output:
%   X: updated n x n matrix
%   Z: updated n x (n-r) matrix
% =========================================================================
% Implemented by Qilun Luo, Oct 22, 2023

    [~, n] = size(A);
    I = eye(n);
    
    L0 = norm(A*Z, 'fro')^2 + 1/2/rho*(norm((X - I + Z*Z')*rho +L, 'fro')^2 - norm(L, 'fro')^2);
    if(L0 > Gamma)

        casenum = 2;
        switch casenum
            case 1
                Z = eye(n, n-r);
            case 2
                ind = randsample(n, n-r);
                sub = sub2ind([n, n-r], ind', 1:n-r);
                Z = zeros(n, n-r);
                Z(sub) = 1;
        end

        X = I - Z*Z';
    end
    
    max_iter = 10;
    for t = 1:max_iter
        %% Update Z
        S = A'*A  - rho*(I-X-L/rho);
        [V, d] = eig(S, 'vector');
        [~, sorted_indices] = sort(d);
        Z = V(:, sorted_indices(1:n-r));

    
        %% Update X
        X = max(I-Z*Z'-L/rho, 0);

    
        %% Stop criteria
        E = norm( (I-Z*Z')*S*Z, 'fro');
        if(E<eps)
            break;
        end
    end

    if(t==max_iter)
        disp('BCD not converge!')
    end
    
end