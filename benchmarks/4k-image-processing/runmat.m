seed_default = 0;
B_default = 16; H_default = 2160; W_default = 3840;
gain_default = single(1.0123);
bias_default = single(-0.02);
gamma_default = single(1.8);
eps0_default = single(1e-6);

if ~exist('seed','var'), seed = seed_default; end
rng(seed);

if ~exist('B','var'), B = B_default; end
if ~exist('H','var'), H = H_default; end
if ~exist('W','var'), W = W_default; end
if ~exist('gain','var'), gain = gain_default; else gain = single(gain); end
if ~exist('bias','var'), bias = bias_default; else bias = single(bias); end
if ~exist('gamma','var'), gamma = gamma_default; else gamma = single(gamma); end
if ~exist('eps0','var'), eps0 = eps0_default; else eps0 = single(eps0); end

% Deterministic, pseudo-random field (LCG-based) - parity: compute in double, then cast
bid = reshape(0:B-1, [B 1 1]);
yid = reshape(0:H-1, [1 H 1]);
xid = reshape(0:W-1, [1 1 W]);
strideHW = H * W;
strideW = W;
seed32 = double(seed);
idx = bid .* strideHW + yid .* strideW + xid + seed32;
state = mod(1664525 .* idx + 1013904223, 4294967296.0);
imgs = single(state) ./ single(4294967296.0);

% Use nested means to ensure exact MATLAB semantics for dims [2 3]
mu = mean(mean(imgs, 2), 3);
sigma = sqrt(mean(mean((imgs - mu).^2, 2), 3) + eps0);

out = ((imgs - mu) ./ sigma) * gain + bias;
% Clamp to avoid NaNs from fractional power on negatives
out = max(out, single(0));
out = out .^ gamma;
mse = mean((out - imgs).^2, 'all');

fprintf('RESULT_ok MSE=%.6e\n', double(mse));