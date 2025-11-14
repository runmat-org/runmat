seed_default = 0;
B_default = 16; H_default = 2160; W_default = 3840;
gain_default = single(1.0123);
bias_default = single(-0.02);
gamma_default = single(1.8);
eps0_default = single(1e-6);

if ~exist('seed','var'), seed = seed_default; end
rng(seed);

env_B = getenv('IMG_B');
if numel(env_B)
  B_default = str2double(env_B);
end
env_H = getenv('IMG_H');
if numel(env_H)
  H_default = str2double(env_H);
end
env_W = getenv('IMG_W');
if numel(env_W)
  W_default = str2double(env_W);
end

if ~exist('B','var'), B = B_default; end
if ~exist('H','var'), H = H_default; end
if ~exist('W','var'), W = W_default; end
if ~exist('gain','var'), gain = gain_default; else gain = single(gain); end
if ~exist('bias','var'), bias = bias_default; else bias = single(bias); end
if ~exist('gamma','var'), gamma = gamma_default; else gamma = single(gamma); end
if ~exist('eps0','var'), eps0 = eps0_default; else eps0 = single(eps0); end

use_gpu = exist('gpuArray', 'builtin') || exist('gpuArray', 'file');
if use_gpu
  imgs = gpuArray(rand(B, H, W, 'single'));
  gain = gpuArray(gain);
  bias = gpuArray(bias);
  gamma = gpuArray(gamma);
  eps0 = gpuArray(eps0);
else
  imgs = rand(B, H, W, 'single');
end

% Reduce over dims 2 and 3 in a single call, keeping native precision
mu = single(mean(imgs, [2 3], 'native'));
sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));

out = single(((imgs - mu) ./ sigma) * gain + bias);
% Clamp to avoid NaNs from fractional power on negatives
if use_gpu
  zero_scalar = gpuArray(single(0));
else
  zero_scalar = single(0);
end
out = max(out, zero_scalar);
out = single(out .^ gamma);
if use_gpu
  mse = gather(mean((out - imgs).^2, 'all'));
else
  mse = mean((out - imgs).^2, 'all');
end

fprintf('RESULT_ok MSE=%.6e\n', double(mse));