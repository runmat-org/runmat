seed_default = 0;
p_default = 128;
C_default = 2048;
T_default = 200;
mu_default = single(0.5);
eps0_default = single(1e-3);

if ~exist('seed','var'), seed = seed_default; end
rng(seed);

if ~exist('p','var'), p = p_default; end
if ~exist('C','var'), C = C_default; end
if ~exist('T','var'), T = T_default; end
if ~exist('mu','var'), mu = mu_default; else mu = single(mu); end
if ~exist('eps0','var'), eps0 = eps0_default; else eps0 = single(eps0); end

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