rng(0);
n = 200_000; d = 1024; k = 8; iters = 15;
A = rand(n, d, 'single');
mu = mean(A, 1, 'omitnan');
A = A - mu;
G = (A' * A) / single(n - 1);

Q = rand(d, k, 'single');
for j = 1:k
  nj = norm(Q(:, j));
  if nj > 0
    Q(:, j) = Q(:, j) ./ nj;
  end
end

for t = 1:iters
  Q = G * Q;
  for j = 1:k
    nj = norm(Q(:, j));
    if nj > 0
      Q(:, j) = Q(:, j) ./ nj;
    end
  end
end

Lambda = diag(Q' * G * Q);
explained = double(Lambda) / sum(double(diag(G)));
fprintf('RESULT_ok EXPLAINED1=%.4f TOPK_SUM=%.6e\n', explained(1), sum(double(Lambda)));