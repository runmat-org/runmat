seed_default = 0;
M_default = 1000000;
T_default = 256;
S0_default = single(100);
mu_default = single(0.05);
sigma_default = single(0.20);
dt_default = single(1.0 / 252.0);
K_default = single(100.0);

if ~exist('seed','var'), seed = seed_default; end
rng(seed);

if ~exist('M','var')
  env_M = getenv('MC_M');
  if numel(env_M)
    M_default = str2double(env_M);
  end
end
if ~exist('T','var')
  env_T = getenv('MC_T');
  if numel(env_T)
    T_default = str2double(env_T);
  end
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

for t = 1:T
  Z = randn(M, 1, 'single');
  S = S .* exp(drift + scale .* Z);
end

payoff = max(S - K, 0);
price  = mean(payoff, 'all') * exp(-mu * T * dt);
price_str = char(num2str(double(price)));
disp(['RESULT_ok PRICE=' price_str]);