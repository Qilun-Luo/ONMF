function C = findConnComp(K, r)
%findConnComp for finding the connected component of the graph.
% =========================================================================
%  Input:
%   K: K = I-ZZ^T the representation of rank space of A
%   r: number of component
%  Output:
%   C: C is of n x r s.t. K = CC^T
% =========================================================================
% Implemented by Qilun Luo, Nov. 05, 2023

    n = size(K, 1);

    % Define initial thresholds
    min_threshold = 1e-10;
    max_threshold = 1e-1;
    tol = (min_threshold + max_threshold) / 2;
    
    max_iterations = 100;  % to avoid infinite loops
    iteration = 0;

    while iteration < max_iterations
        K_adj = K;
        K_adj(abs(K_adj) < tol) = 0;
        
        G_adj = K_adj;
        G_adj(G_adj~=0) = 1;
        G_adj = graph((G_adj+G_adj')/2);
        num_components = max(conncomp(G_adj));
        
        if num_components == r
            break;
        elseif num_components > r
            max_threshold = tol;
        else
            min_threshold = tol;
        end
        
        tol = (min_threshold + max_threshold) / 2;
        iteration = iteration + 1;
    end
    if(iteration==max_iterations)
        disp('finding connected component fails')
    end
    G = G_adj;
    bins = conncomp(G);
    binnodes = accumarray(bins', 1:numel(bins), [], @(v) {sort(v')});
    Ki = cell(length(binnodes), 1);
    C = zeros(n, r);
    for binidx = 1:numel(binnodes)
        Ki{binidx} = K(binnodes{binidx},binnodes{binidx});
        C(binnodes{binidx}, binidx) = Ki{binidx}(:,1)/norm(Ki{binidx}(:,1));
    end
    C = C(:,1:r);
end