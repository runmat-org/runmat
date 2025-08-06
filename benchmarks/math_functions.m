% Mathematical Functions Benchmark
% Tests performance of common mathematical operations

function run_math_benchmark()
    fprintf('Running mathematical functions benchmark...\n');
    
    % Test data sizes
    sizes = [10000, 100000, 1000000];
    
    for i = 1:length(sizes)
        n = sizes(i);
        fprintf('\n=== Array size: %d elements ===\n', n);
        
        % Generate test data
        x = randn(1, n);
        y = randn(1, n);
        
        % Trigonometric functions
        tic;
        result1 = sin(x);
        sin_time = toc;
        fprintf('sin(): %.4f seconds\n', sin_time);
        
        tic;
        result2 = cos(x);
        cos_time = toc;
        fprintf('cos(): %.4f seconds\n', cos_time);
        
        tic;
        result3 = tan(x);
        tan_time = toc;
        fprintf('tan(): %.4f seconds\n', tan_time);
        
        % Exponential and logarithmic
        x_pos = abs(x) + 1; % Ensure positive for log
        
        tic;
        result4 = exp(x);
        exp_time = toc;
        fprintf('exp(): %.4f seconds\n', exp_time);
        
        tic;
        result5 = log(x_pos);
        log_time = toc;
        fprintf('log(): %.4f seconds\n', log_time);
        
        tic;
        result6 = sqrt(x_pos);
        sqrt_time = toc;
        fprintf('sqrt(): %.4f seconds\n', sqrt_time);
        
        % Statistical functions
        tic;
        result7 = sum(x);
        sum_time = toc;
        fprintf('sum(): %.4f seconds\n', sum_time);
        
        tic;
        result8 = mean(x);
        mean_time = toc;
        fprintf('mean(): %.4f seconds\n', mean_time);
        
        tic;
        result9 = std(x);
        std_time = toc;
        fprintf('std(): %.4f seconds\n', std_time);
    end
end

% Run the benchmark
run_math_benchmark();