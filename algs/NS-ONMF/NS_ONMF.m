function [Z, Out] = NS_ONMF(A, opts)
%ONMF for finding null space of A over the Stiefel Manifold.
% =========================================================================
%  Input:
%   A: the given non-negative matrix of size m x n
%   opts: optional parameters
%  Output:
%   Z: null space basis of A
%   Out: Other output information
% =========================================================================
% Implemented by Qilun Luo, Oct 22, 2023

    %% Parameter settings
    max_iter = 100;
    epsilon = 1e-6;
    flag_debug = 1;
    r = 1;
    rho = 1e-10;
    rhomax = 1e8;
    gamma = 1.2;
    tau = 0.1;
    eta = 0.999;
    BCD_MIter = 10;
    
    if ~exist('opts', 'var')
        opts = [];
    end
    if  isfield(opts, 'max_iter');      max_iter = opts.max_iter;       end
    if  isfield(opts, 'BCD_MIter');     BCD_MIter = opts.BCD_MIter;     end
    if  isfield(opts, 'epsilon');       epsilon = opts.epsilon;         end
    if  isfield(opts, 'flag_debug');    flag_debug = opts.flag_debug;   end
    if  isfield(opts, 'r');             r = opts.r;                     end
    if  isfield(opts, 'rho');           rho = opts.rho;                 end
    if  isfield(opts, 'rhomax');        rhomax = opts.rhomax;           end
    if  isfield(opts, 'gamma');         gamma = opts.gamma;             end
    if  isfield(opts, 'tau');           tau = opts.tau;                 end
    if  isfield(opts, 'eta');           eta = opts.eta;                 end
    if  isfield(opts, 'Agt');           Agt = opts.Agt;                 end
    
    [~, n] = size(A);
    
    I = eye(n);

    %% Initial
    casenum = 1;
    switch casenum
        case 1
            Z = eye(n, n-r);
        case 2
            sub = randsample(n, n-r);
            ind = sub2ind([n, n-r], sub', 1:n-r);
            Z = zeros(n, n-r);
            Z(ind) = 1;
        case 3
            [~,~,V] = svds(A, r);
            [Z, ~, ~] = svds(eye(n) - V*V', n-r);
        case 4
            [V, D] = eig(full(A'*A));
            [~, sorted_indices] = sort(diag(D));
            Z = V(:, sorted_indices(r:n));
    end
    X = I - Z*Z'; 



    L = zeros(n, n);

    
    Out = [];
    Out.obj = [];
    Out.nrmC = [];
    Out.relError = [];
    Out.cpu = [];
    if isfield(opts, 'Agt')
        Out.cpu = [Out.cpu; 0];
        C = eye(n, r);
        B = A*C;
        relError = norm(Agt - B*C')/norm(Agt, 'fro');
        Out.relError = [Out.relError; relError];
        tic
    end

    Lfun = norm(A*Z, 'fro')^2/2 + 1/2/rho*(norm((X - I + Z*Z')*rho +L, 'fro')^2  - norm(L, 'fro')^2);
    Gamma = max(norm(A*Z, 'fro')^2, Lfun);

    Out.Gamma = Gamma;
    
    for k = 1:max_iter
        X0 = X;
        Z0 = Z;
    
        %% BCD Algorithm to solve for W = (X, Y, Z)
        [X, Z] = BCD_solver(A, r, X0, Z0, rho, L, 1/k, Gamma, BCD_MIter);

        %% Convergence Check
        Lfun = norm(A*Z, 'fro')^2/2 + 1/2/rho*(norm((X - I + Z*Z')*rho +L, 'fro')^2  - norm(L, 'fro')^2);

        nrmC = norm(X-I+Z*Z', 'fro');

        if flag_debug
            fprintf(['Iter = %d\tObj = %.6f\t||c(W)|| = %.6e\trho = %.6e\n'], k, Lfun,  nrmC, rho);
        end

        Out.obj = [Out.obj; Lfun];
        Out.nrmC = [Out.nrmC; nrmC];

        rho_n = max(rho*gamma, norm(L, 'fro')^(1+tau));

        if nrmC<epsilon || rho_n > rhomax
            break;
        end

        %% Update multipliers
        L = L + rho*(X-I+Z*Z');

        %% Update rho
        if k > 1
            C = norm(X-I+Z*Z', 'fro');
            C0 = norm(X0-I+Z0*Z0', 'fro');
            if (C > eta*C0)
                rho = rho_n;
            end
        end

        if isfield(opts, 'Agt')
            Out.cpu = [Out.cpu; Out.cpu(end)+toc];
            K = I-Z*Z';
            C = findConnComp(K, r);
            B = A*C;
            relError = norm(Agt - B*C')/norm(Agt, 'fro');
            Out.relError = [Out.relError; relError];
            tic
        end
    end

    K = I-Z*Z';
    C = findConnComp(K, r);
    B = A*C;

    Out.B = B;
    Out.C = C;
    Out.K = K;
end

