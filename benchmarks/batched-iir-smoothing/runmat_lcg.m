seed_default = 0;
tshirt_env = getenv('IIR_TSHIRT');
if double(isempty(tshirt_env))
  tshirt_size = 'xs';
else
  tshirt_size = lower(strtrim(tshirt_env));
end

xs_match = sum(double(strcmp(tshirt_size, {'xs', 'xsmall', 'x-small'})));
s_match = sum(double(strcmp(tshirt_size, {'s', 'small'})));
m_match = sum(double(strcmp(tshirt_size, {'m', 'medium'})));
l_match = sum(double(strcmp(tshirt_size, {'l', 'large'})));
xl_lite_match = sum(double(strcmp(tshirt_size, {'xl-lite', 'xllite', 'xl_lite'})));
xl_mid_match = sum(double(strcmp(tshirt_size, {'xl-mid', 'xlmid', 'xl_mid'})));
xl_match = sum(double(strcmp(tshirt_size, {'xl', 'xlarge', 'full'})));

if xs_match
  M_default = 2048;
  T_default = 16;
elseif s_match
  M_default = 32768;
  T_default = 128;
elseif m_match
  M_default = 131072;
  T_default = 512;
elseif l_match
  M_default = 524288;
  T_default = 2048;
elseif xl_lite_match
  M_default = 1000000;
  T_default = 1536;
elseif xl_mid_match
  M_default = 1500000;
  T_default = 3072;
elseif xl_match
  M_default = 2000000;
  T_default = 4096;
else
  error('IIR_TSHIRT=%s is invalid (expected xs|s|m|l|xl-lite|xl-mid|xl/full)', tshirt_env);
end

if ~exist('seed','var'), seed = seed_default; end
rng(seed);

if ~exist('M','var'), M = M_default; end
if ~exist('T','var'), T = T_default; end
if ~exist('alpha','var'), alpha = single(0.98); else, alpha = single(alpha); end
if ~exist('beta','var'), beta = single(0.02); else, beta = single(beta); end

fprintf('CONFIG seed=%d M=%d T=%d alpha=%g beta=%g tshirt=%s\n', seed, M, T, double(alpha), double(beta), tshirt_size);

Y = zeros(M, 1, 'single');
rid = reshape(0:M-1, [M 1]);
seed_shift = double(seed);
for t = 1:T
  salt = double(t - 1) * double(M);
  idx = rid + salt + seed_shift;
  state = mod(1664525.0 .* idx + 1013904223.0, 4294967296.0);
  x = single(state ./ 4294967296.0);
  Y = alpha .* Y + beta .* x;
end

mean_y = mean(Y);
fprintf('RESULT_ok MEAN=%.6e\n', double(mean_y));