% SSA Optimization Benchmark
% Tests patterns that benefit from SSA optimizations:
% - Constant folding
% - Common subexpression elimination (CSE)
% - Loop-invariant code motion (LICM)

N = 1000000;
ITERS = 10;

fprintf('SSA Optimization Benchmark\n');
fprintf('N = %d, ITERS = %d\n\n', N, ITERS);

% =========================================================================
% Test 1: Constant Folding
% =========================================================================
% Pattern: repeated use of compile-time constants
% SSA benefit: constants folded, no runtime computation

x = zeros(N, 1);
tic;
for iter = 1:ITERS
    a = 2.5;
    b = 3.5;
    c = a + b;        % Should fold to 6.0
    d = c * 2.0;      % Should fold to 12.0
    e = d - 4.0;      % Should fold to 8.0
    for i = 1:N
        x(i) = e * i; % Only multiply by constant 8.0
    end
end
t1 = toc;
fprintf('Test 1 (Constant Folding): %.3f ms, sum=%.2f\n', t1*1000, sum(x));

% =========================================================================
% Test 2: Common Subexpression Elimination
% =========================================================================
% Pattern: same expression computed multiple times
% SSA benefit: compute once, reuse result

a = rand(N, 1);
b = rand(N, 1);
result = zeros(N, 1);

tic;
for iter = 1:ITERS
    for i = 1:N
        % Redundant: a(i) + b(i) computed twice
        t1_val = a(i) + b(i);
        t2_val = a(i) + b(i);  % CSE should eliminate
        result(i) = t1_val * t2_val;
    end
end
t2 = toc;
fprintf('Test 2 (CSE): %.3f ms, sum=%.6f\n', t2*1000, sum(result));

% =========================================================================
% Test 3: Loop-Invariant Code Motion
% =========================================================================
% Pattern: computation inside loop that doesn't depend on loop variable
% SSA benefit: hoist to preheader, compute once

c1 = 3.14159;
c2 = 2.71828;
result2 = zeros(N, 1);

tic;
for iter = 1:ITERS
    for i = 1:N
        % Loop-invariant: c1 * c2 doesn't depend on i
        inv = c1 * c2;  % LICM should hoist this
        result2(i) = inv * i;
    end
end
t3 = toc;
fprintf('Test 3 (LICM): %.3f ms, sum=%.6f\n', t3*1000, sum(result2));

% =========================================================================
% Test 4: Combined (realistic pattern)
% =========================================================================
% Monte Carlo-style loop with mixed optimization opportunities

prices = rand(N, 1) * 100 + 50;  % Random prices 50-150
strike = 100.0;
rate = 0.05;
time = 1.0;
discount = exp(-rate * time);  % Loop-invariant

payoffs = zeros(N, 1);
tic;
for iter = 1:ITERS
    for i = 1:N
        % Constant folding: rate * time
        % LICM: discount computation
        % CSE: if used multiple times
        df = exp(-rate * time);  % Should be hoisted/folded
        payoff = max(prices(i) - strike, 0);
        payoffs(i) = payoff * df;
    end
end
t4 = toc;
fprintf('Test 4 (Combined): %.3f ms, mean=%.6f\n', t4*1000, mean(payoffs));

fprintf('\nTotal: %.3f ms\n', (t1+t2+t3+t4)*1000);
fprintf('RESULT_ok TOTAL=%.3f\n', (t1+t2+t3+t4)*1000);
