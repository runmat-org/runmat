seed_default = 0;
p_default = 128;
C_default = 2048;
T_default = 200;
mu_default = single(0.5);
eps0_default = single(1e-3);

% Optional t-shirt overrides for harness use
nlms_tshirt = strtrim(lower(getenv('NLMS_TSHIRT')));
has_tshirt = length(nlms_tshirt);
if has_tshirt
  s_match = sum(double(strcmp(nlms_tshirt, {'s', 'small'})));
  m_match = sum(double(strcmp(nlms_tshirt, {'m', 'medium'})));
  l_match = sum(double(strcmp(nlms_tshirt, {'l', 'large'})));
  if s_match
    p_default = 128;
    C_default = 512;
    T_default = 200;
  elseif m_match
    p_default = 128;
    C_default = 2048;
    T_default = 200;
  elseif l_match
    p_default = 256;
    C_default = 4096;
    T_default = 200;
  else
    warning('NLMS_TSHIRT=%s is not recognized; using defaults', nlms_tshirt);
  end
end

% Direct env overrides for scalar params (e.g., NLMS_C=512)
env_p = getenv('NLMS_P');
if numel(env_p)
  p_default = str2double(env_p);
end
env_c = getenv('NLMS_C');
if numel(env_c)
  C_default = str2double(env_c);
end
env_t = getenv('NLMS_T');
if numel(env_t)
  T_default = str2double(env_t);
end

if ~exist('seed','var'), seed = seed_default; end
rng(seed);

if ~exist('p','var'), p = p_default; end
if ~exist('C','var'), C = C_default; end
if ~exist('T','var'), T = T_default; end
if ~exist('mu','var'), mu = mu_default; else mu = single(mu); end
if ~exist('eps0','var'), eps0 = eps0_default; else eps0 = single(eps0); end
if ~exist('debug_gpu_mse','var'), debug_gpu_mse = 0; end

W = zeros(p, C, 'single');

for t = 1:T
  x = rand(p, C, 'single');
  d = sum(x .* x, 1, 'native');
  y = sum(x .* W, 1, 'native');
  e = d - y;
  nx = d + eps0;
  scale = repmat(single(e ./ nx), p, 1);
  W = W + mu * (x .* scale);
end

d_single = single(d);
y_single = single(y);
err_single = d_single - y_single;
gpu_mse = mean(err_single .* err_single, 'all', 'native');

final_d = sum(x .* x, 1, 'native');
final_y = sum(x .* W, 1, 'native');
mse = mean((final_d - final_y).^2, 'all', 'native');

if debug_gpu_mse
  fprintf('gpu_mse=%.6e mse_ref=%.6e delta=%.3e\n', gpu_mse, mse, abs(double(mse) - double(gpu_mse)));
end

fprintf('RESULT_ok MSE=%.6e\n', mse);

