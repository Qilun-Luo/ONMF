function X = update_X(H, tau)

[U, S, V] = svd(H, 'econ');
s = diag(S);
S = diag(max(s-tau, 0));
X = U*S*V';
