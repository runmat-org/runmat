% Quick SSA Optimization Benchmark
% Use assignments only (no disp/fprintf) to ensure JIT path is taken
% Time externally with PowerShell: Measure-Command { runmat ... }

% Test 1: Constant folding opportunity
for i = 1:50000
    a = 2.5 + 3.5;
    b = a * 2.0;
    c = b - 4.0;
end

% Test 2: Variable arithmetic loop  
x = 1.0;
for i = 1:100000
    x = x + 0.00001;
end

% Test 3: CSE opportunity (a*b appears twice per iteration)
a = 3.0;
b = 4.0;
for i = 1:50000
    t = a * b + 2.0;
    u = a * b - 1.0;
    v = t + u;
end

% Test 4: LICM opportunity (k*k should be hoisted out of loop)
k = 7.0;
y = 0.0;
for i = 1:50000
    y = y + k * k;
end

% Use scalar addition instead of matrix to avoid CreateMatrixDynamic
result = c + x + v + y;
