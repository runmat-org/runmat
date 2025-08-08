% Mathematical Functions Benchmark
% Tests performance of common mathematical operations on large arrays

function run_math_benchmark()
    fprintf('=== Mathematical Functions Benchmark ===\n');
    
    % Test multiple array sizes
    sizes = [50000, 200000, 500000];
    for s = 1:length(sizes)
        n = sizes(s);
        fprintf('\nArray size: %d elements\n', n);

        % Generate test data
        x = randn(1, n);
        x_pos = abs(x) + 1; % Ensure positive for log/sqrt

        % Trigonometric functions (test JIT mathematical implementations)
        tic; result_sin = sin(x); sin_time = toc;
        fprintf('  sin() time: %.6f\n', sin_time);

        tic; result_cos = cos(x); cos_time = toc;
        fprintf('  cos() time: %.6f\n', cos_time);

        tic; result_exp = exp(x); exp_time = toc;
        fprintf('  exp() time: %.6f\n', exp_time);

        tic; result_log = log(x_pos); log_time = toc;
        fprintf('  log() time: %.6f\n', log_time);

        tic; result_sqrt = sqrt(x_pos); sqrt_time = toc;
        fprintf('  sqrt() time: %.6f\n', sqrt_time);

        % Statistical functions (test runtime builtin integration)
        tic; result_sum = sum(x); sum_time = toc;
        fprintf('  sum() time: %.6f\n', sum_time);

        tic; result_mean = mean(x); mean_time = toc;
        fprintf('  mean() time: %.6f\n', mean_time);

        tic; result_std = std(x); std_time = toc;
        fprintf('  std() time: %.6f\n', std_time);
    end
    
    fprintf('\nMathematical functions benchmark completed.\n');
end

% Run the benchmark
run_math_benchmark();