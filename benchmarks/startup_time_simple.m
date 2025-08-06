% Simple Startup Time Benchmark
% Measures cold start performance without function definitions

fprintf('Running startup time benchmark...');
fprintf('Note: This measures execution of a simple script');
fprintf('For true startup time, measure from process launch to first output');

% Simple operations to test startup overhead
tic;

% Basic arithmetic
a = 42;
b = 58;
c = a + b;

% Simple matrix operation
m = [1, 2; 3, 4];
result = m * 2;

% Basic function call
d = sin(3.14159 / 2);

execution_time = toc;

fprintf('Simple script execution time: ');
fprintf('seconds');
fprintf('Result verification: c = ');
fprintf(', d = ');