% Intensive SSA Optimization Benchmark
% This benchmark runs longer to amortize startup overhead

% Test: Pure arithmetic loops (most likely to benefit from JIT)
x = 0.0;
for i = 1:1000000
    x = x + 1.0;
end

% Store result to prevent DCE
result = x;
