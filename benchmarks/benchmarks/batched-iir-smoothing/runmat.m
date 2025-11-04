rng(0);
if ~exist('M','var'), M = 2000000; end
if ~exist('T','var'), T = 4096; end
alpha = single(0.98); beta = single(0.02);

Y = zeros(M, 1, 'single');
for t = 1:T
  x = rand(M, 1, 'single');
  Y = alpha .* Y + beta .* x;
end

mean_y = mean(Y);
fprintf('RESULT_ok MEAN=%.6e\n', double(mean_y));