rng(0);
p = 128; C = 2048; T = 200;
mu = single(0.5); eps0 = single(1e-3);

W = zeros(p, C, 'single');

for t = 1:T
  x = rand(p, C, 'single');
  d = sum(x .* x, 1);              % desired: squared norm per column
  y = sum(x .* W, 1);              % current estimate per column
  e = d - y;                       % error per column
  nx = sum(x .^ 2, 1) + eps0;      % normalization term per column
  W = W + mu * x .* (e ./ nx);     % NLMS update (broadcast over rows)
end

mse = mean((d - sum(x .* W, 1)).^2, 'all');
fprintf('RESULT_ok MSE=%.6e\n', double(mse));