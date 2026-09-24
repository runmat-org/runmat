% Official v0.6.2: benchmark --jit --iterations 12 fails on measured iteration 7.
% CPU acceleration disabled; no fixture I/O, timing, printing or string dispatch.
z = sin((1:1000)');
M = numel(z);
payoff = zeros(M, 1);
for i = 1:M
    payoff(i) = exp(-0.03) * max(100 * exp(0.01 + 0.2*z(i)) - 100, 0);
end
price = mean(payoff);
stderr_value = std(payoff) / sqrt(M);
