rng(0);
B = 16; H = 1024; W = 1024;

raw = rand(B, H, W, 'single');

dark   = 0.02 + 0.01 * rand(H, W, 'single');      % dark frame map
ffc    = 0.98 + 0.04 * rand(H, W, 'single');      % flat-field correction map
gain   = 1.50 + 0.50 * rand(H, W, 'single');      % radiometric gain
offset = -0.05 + 0.10 * rand(H, W, 'single');     % radiometric offset

lin = (raw - dark) .* ffc;                        % dark/flat correction
radiance = lin .* gain + offset;                  % radiometric calibration
radiance = max(radiance, 0);                      % clamp negatives

tempK = 273.15 + 80 * log1p(radiance);           % toy temp conversion

mean_temp = mean(tempK, 'all');
fprintf('RESULT_ok MEAN_TEMP=%.6f\n', double(mean_temp));