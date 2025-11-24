function [K, B, Out] = sn_onmf(A, opts)
% ORTHOGONAL NONNEGATIVE MATRIX FACTORIZATION BY SPARSITY AND NUCLEAR NORM
%  OPTIMIZATION
% Ref: Pan, J., & Ng, M. K. (2018). Orthogonal nonnegative matrix 
% factorization by sparsity and nuclear norm optimization. 
% SIAM Journal on Matrix Analysis and Applications, 39(2), 856-875.
% =========================================================================
%  Input:
%   A: Given non-negative matrix
%   opts: Optional parameters
%  Output:
%   K: Projection matrix K=BB^T
%   Out: Other output information
% =========================================================================
% Implemented by QL, July 7, 2023

%% Parameter settings
max_iter = 100;
epsilon = 1e-6;
flag_debug = 1;
theta = 1e-4;
beta = 1e-7;
mu1 = 0.1;
mu2 = 0.1;
mu3 = 0.1;
r = 1;
eta = 1.2;
max_eta =  1e10;

if ~exist('opts', 'var')
    opts = [];
end  
if  isfield(opts, 'max_iter');      max_iter = opts.max_iter;       end
if  isfield(opts, 'epsilon');       epsilon = opts.epsilon;         end
if  isfield(opts, 'flag_debug');    flag_debug = opts.flag_debug;   end
if  isfield(opts, 'r');             r = opts.r;                     end
if  isfield(opts, 'theta');         theta = opts.theta;             end
if  isfield(opts, 'beta');          beta = opts.beta;               end
if  isfield(opts, 'mu1');           mu1 = opts.mu1;                 end
if  isfield(opts, 'mu2');           mu2 = opts.mu2;                 end
if  isfield(opts, 'mu3');           mu3 = opts.mu3;                 end
if  isfield(opts, 'eta');           eta = opts.eta;                 end
if  isfield(opts, 'Agt');           Agt = opts.Agt;                 end

%% Initialization
[m, n] = size(A);
Im = eye(m);


B = rand(m, r);
K = B*B'/m/r;
% K = eye(m)/m/r;
Y = K;
X = K;
Z = K;
L1 = zeros(m,m);
L2 = zeros(m,m);
L3 = zeros(m,m);

Out.relError = [];
Out.cpu = [];
if isfield(opts, 'Agt')
    Out.cpu = [Out.cpu; 0];
    C = A'*B;
    relError = norm(Agt - C*B')/norm(Agt, 'fro');
    Out.relError = [Out.relError; relError];
    tic
end

for t = 1:max_iter
    K_old = K;
    K = (A*A'+mu1*Y+L1+mu2*X+L2+mu3*Z+L3)/(A*A'+(mu1+mu2+mu3)*Im);
    
    Y = ((K-L1/mu1)'+(K-L1/mu1))/2;
    L1 = L1 + mu1*(Y-K);

    X = update_X(K-L2/mu2, beta/mu2);
    L2 = L2 + mu2*(X-K);

    W = K-L3/mu3;
    Z = max(0, W-theta/mu3);

    L3 = L3 + mu3*(Z-K);

    nor1 = norm(K_old - K, 'fro');
    nor2 = norm(X - K, 'fro');
    nor3 = norm(Z - K, 'fro');
    Out.nor1 = nor1;
    Out.nor2 = nor2;
    Out.nor3 = nor3;
    if flag_debug
        fprintf('iter = %d, nor1 = %.6f, nor2 = %.6f, nor3 = %.6f\n', t, nor1, nor2, nor3);
    end
    if(max([nor1, nor2, nor3])<epsilon)
        break;
    end
    mu1 = min(eta*mu1, max_eta);
    mu2 = min(eta*mu2, max_eta);
    mu3 = min(eta*mu3, max_eta);

    if isfield(opts, 'Agt')
        Out.cpu = [Out.cpu; Out.cpu(end)+toc];
        [V, D] = eig(K);
        [~, ind] = sort(diag(D), 'descend');
        B = V(:, ind(1:r));
        C = A'*B;
        relError = norm(Agt - C*B')/norm(Agt, 'fro');
        Out.relError = [Out.relError; relError];
        tic
    end
end


% Compute B
K = (K+K')/2;
[V, D] = eig(K);
[~, ind] = sort(diag(D), 'descend');
B = V(:, ind(1:r));

Out.B = B;
