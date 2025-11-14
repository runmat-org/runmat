seed_default = 0;
M_default = 1000000;
T_default = 256;
S0_default = single(100);
mu_default = single(0.05);
sigma_default = single(0.20);
dt_default = single(1.0 / 252.0);
K_default = single(100.0);

if ~exist('seed','var'), seed = seed_default; end
env_M = getenv('MC_M');
if numel(env_M)
  M_default = str2double(env_M);
end
env_T = getenv('MC_T');
if numel(env_T)
  T_default = str2double(env_T);
end

if ~exist('M','var'), M = M_default; end
if ~exist('T','var'), T = T_default; end
if ~exist('S0','var'), S0 = S0_default; else S0 = single(S0); end
if ~exist('mu','var'), mu = mu_default; else mu = single(mu); end
if ~exist('sigma','var'), sigma = sigma_default; else sigma = single(sigma); end
if ~exist('dt','var'), dt = dt_default; else dt = single(dt); end
if ~exist('K','var'), K = K_default; else K = single(K); end

S = ones(M, 1, 'single') * S0;
sqrt_dt = sqrt(dt);
drift = (mu - 0.5 * sigma^2) * dt;
scale = sigma * sqrt_dt;

rid = reshape(0:M-1, [M 1]);
seed_shift = double(seed);
twoM = double(M) * 2.0;
for t = 1:T
  salt = double(t - 1) * twoM;
  idx1 = double(rid) + salt + seed_shift;
  idx2 = double(rid) + salt + double(M) + seed_shift;
  state1 = mod(1664525.0 .* idx1 + 1013904223.0, 4294967296.0);
  state2 = mod(1664525.0 .* idx2 + 1013904223.0, 4294967296.0);
  u1 = max(state1 ./ 4294967296.0, 1.0 / 4294967296.0);
  u2 = state2 ./ 4294967296.0;
  r = sqrt(-2.0 .* log(u1));
  theta = 2.0 * pi .* u2;
  z = single(r .* cos(theta));
  S = S .* exp(drift + scale .* z);
end

payoff = max(S - K, 0);
price  = mean(payoff, 'all') * exp(-mu * T * dt);
fprintf('RESULT_ok PRICE=%.6f\n', double(price));