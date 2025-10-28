rng(0);
B = 16; H = 2160; W = 3840;
gain = single(1.0123); bias = single(-0.02); gamma = single(1.8); eps0 = single(1e-6);

imgs = rand(B, H, W, 'single');

mu = mean(imgs, [2 3]);
sigma = sqrt(mean((imgs - mu).^2, [2 3]) + eps0);

out = ((imgs - mu) ./ sigma) * gain + bias;
out = out .^ gamma;
mse = mean((out - imgs).^2, 'all');

fprintf('RESULT_ok MSE=%.6e\n', double(mse));