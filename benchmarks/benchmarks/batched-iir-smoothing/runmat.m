rng(0);
M = 2_000_000; T = 4096;
alpha = single(0.98); beta = single(0.02);

X = rand(M, T, 'single');
Y = zeros(M, 1, 'single');

for t = 1:T
  Y = alpha .* Y + beta .* X(:, t);
end

mean_y = mean(Y);
fprintf('RESULT_ok MEAN=%.6e\n', double(mean_y));