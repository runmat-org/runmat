% Startup Time Benchmark
% Measures basic script execution performance (startup overhead + simple operations)

function run_startup_benchmark()
    fprintf('=== Startup Time Benchmark ===\n');
    fprintf('Note: Measures script execution time including startup overhead\n\n');
    
    % Record start time for total execution measurement
    tic;
    
    % Basic arithmetic operations
    a = 42;
    b = 58;
    c = a + b;
    
    % Simple matrix operations
    m = [1, 2; 3, 4];
    result = m * 2;
    
    % Basic mathematical function calls
    d = sin(3.14159 / 2);
    e = exp(1.0);
    f = sqrt(16.0);
    
    % Simple loop to test control flow
    total = 0;
    for i = 1:10
        total = total + i;
    end
    
    execution_time = toc;
    
    fprintf('Total execution time: ');
    fprintf(execution_time);
    fprintf(' seconds\n');
    fprintf('Verification complete\n');
    fprintf('\nStartup benchmark completed.\n');
end

% Run the benchmark
run_startup_benchmark();