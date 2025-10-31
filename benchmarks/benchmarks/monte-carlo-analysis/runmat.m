rng(0);
M = 10_000_000; T = 256;
S0 = single(100); mu = single(0.05); sigma = single(0.20);
dt = single(1.0/252.0); K = single(100.0);

S = ones(M, 1, 'single') * S0;
sqrt_dt = sqrt(dt);
drift = (mu - 0.5 * sigma^2) * dt;
scale = sigma * sqrt_dt;

for t = 1:T
  Z = randn(M, 1, 'single');
  S = S .* exp(drift + scale .* Z);
end

payoff = max(S - K, 0);
price  = mean(payoff) * exp(-mu * T * dt);
fprintf('RESULT_ok PRICE=%.6f\n', double(price));