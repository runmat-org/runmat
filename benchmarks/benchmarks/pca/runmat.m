n = 200000; d = 1024; k = 8; iters = 15;

A = gpuArray(rand(n, d));
mu = mean(A, 1);
A = A - mu;
G = (A.' * A) / (n - 1);

Q = gpuArray(rand(d, k));
[Q, R_unused] = qr(Q, 'econ');

for t = 1:iters
  [Q, R_unused] = qr(G * Q, 'econ');
end

Lambda = diag(Q.' * G * Q);
explained = double(Lambda) / sum(double(diag(G)));
fprintf('RESULT_ok EXPLAINED1=%.4f TOPK_SUM=%.6e\n', explained(1), sum(double(Lambda)));