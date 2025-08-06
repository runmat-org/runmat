% Startup Time Benchmark
% Measures cold start performance

function run_startup_benchmark()
    fprintf('Running startup time benchmark...\n');
    fprintf('Note: This measures execution of a simple script\n');
    fprintf('For true startup time, measure from process launch to first output\n\n');
    
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
    
    fprintf('Simple script execution time: %.6f seconds\n', execution_time);
    fprintf('Result verification: c = %d, d = %.4f\n', c, d);
end

% Run the benchmark
run_startup_benchmark();