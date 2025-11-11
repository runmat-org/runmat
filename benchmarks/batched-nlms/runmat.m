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
  % Deterministic pseudo-random design matrix per-iteration (float LCG to match NumPy)
  rid = double(reshape(0:p-1, [p 1]));
  cid = double(reshape(0:C-1, [1 C]));
  salt = double(t - 1) * double(p * C);
  idx = rid .* double(C) + cid + salt + 0.0;
  state = mod(1664525.0 .* idx + 1013904223.0, 4294967296.0);
  x = single(state ./ 4294967296.0);
  % Column-wise dot products
  d = single(dot(x, x, 1));
  y = single(dot(x, W, 1));
  e = d - y;                       % error per column
  nx = d + eps0;  % normalization term per column
  scale = repmat(single(e ./ nx), p, 1);
  W = W + mu * single(double(x) .* double(scale));  % NLMS update on CPU semantics
end

mse = mean((d - single(dot(x, W, 1))).^2, 'all');
fprintf('RESULT_ok MSE=%.6e\n', double(mse));