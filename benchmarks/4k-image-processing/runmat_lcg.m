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
  B_override = str2double(env_B);
  if isfinite(B_override) && B_override > 0 && B_override == floor(B_override)
    B_default = B_override;
  else
    error('IMG_B must be a finite positive integer');
  end
end
env_H = getenv('IMG_H');
if numel(env_H)
  H_override = str2double(env_H);
  if isfinite(H_override) && H_override > 0 && H_override == floor(H_override)
    H_default = H_override;
  else
    error('IMG_H must be a finite positive integer');
  end
end
env_W = getenv('IMG_W');
if numel(env_W)
  W_override = str2double(env_W);
  if isfinite(W_override) && W_override > 0 && W_override == floor(W_override)
    W_default = W_override;
  else
    error('IMG_W must be a finite positive integer');
  end
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
  zero_proto = gpuArray(single(0));
  imgs = zeros(B, H, W, 'like', zero_proto);
  gain = gpuArray(gain);
  bias = gpuArray(bias);
  gamma = gpuArray(gamma);
  eps0 = gpuArray(eps0);
else
  imgs = zeros(B, H, W, 'single');
end

plane = H * W;
row_block = 16;
x_idx = reshape(0:(W - 1), [1 1 W]);
for b = 1:B
  batch_offset = (b - 1) * plane + seed;
  row_start = 1;
  while row_start <= H
    row_count = min(row_block, H - row_start + 1);
    rows = row_start:(row_start + row_count - 1);
    y_idx = reshape((row_start - 1):(row_start + row_count - 2), [1 row_count 1]);
    idx = batch_offset + y_idx * W + x_idx;
    state = mod(1664525 .* idx + 1013904223, 4294967296);
    chunk = single(state) ./ single(4294967296);
    if use_gpu
      imgs(b, rows, :) = gpuArray(chunk);
    else
      imgs(b, rows, :) = chunk;
    end
    row_start = row_start + row_count;
  end
end

% Reduce over dims 2 and 3 in a single call, keeping native precision
mu = single(mean(imgs, [2 3], 'native'));
sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));

out = single(((imgs - mu) ./ sigma) * gain + bias);
% Clamp to avoid NaNs from fractional power on negatives
out = max(out, single(0));
out = single(out .^ gamma);
err = out - imgs;
if use_gpu
  mse = gather(mean(err .* err, 'all'));
else
  mse = mean(err .* err, 'all');
end

fprintf('RESULT_ok MSE=%.6e\n', double(mse));
