% Control Flow Benchmark
% Tests performance of loops, conditionals, and user-defined functions

function run_control_flow_benchmark()
    fprintf('=== Control Flow Benchmark ===\n');

    % Test different iteration counts
    iterations = [1000, 5000, 10000];
    for i = 1:length(iterations)
        n = iterations(i);
        fprintf('\nIteration count: %d\n', n);

        % Test simple for loop with arithmetic
        tic;
        sum_result = 0;
        for j = 1:n
            sum_result = sum_result + j * j;
        end
        loop_time = toc;
        fprintf('  For loop time: %.6f (sum: %d)\n', loop_time, sum_result);

        % Test nested loops
        tic;
        nested_sum = 0;
        loop_size = floor(sqrt(n));
        for outer = 1:loop_size
            for inner = 1:loop_size
                nested_sum = nested_sum + outer * inner;
            end
        end
        nested_time = toc;
        fprintf('  Nested loops time: %.6f (sum: %d)\n', nested_time, nested_sum);

        % Test conditional logic
        tic;
        conditional_count = 0;
        half_n = n/2;
        for k = 1:n
            if k <= half_n
                conditional_count = conditional_count + 1;
            else
                conditional_count = conditional_count + 2;
            end
        end
        conditional_time = toc;
        fprintf('  Conditional time: %.6f (count: %d)\n', conditional_time, conditional_count);

        % Test user-defined function calls
        tic;
        function_result = 0;
        func_iterations = floor(n/10);  % Fewer iterations for function calls
        for l = 1:func_iterations
            function_result = function_result + fibonacci_iterative(10);
        end
        function_time = toc;
        fprintf('  Function calls time: %.6f (result: %d)\n', function_time, function_result);
    end

    fprintf('\nControl flow benchmark completed.\n');
end

% Helper function for testing function call performance
function result = fibonacci_iterative(n)
    if n <= 1
        result = n;
    else
        a = 0;
        b = 1;
        for i = 2:n
            temp = a + b;
            a = b;
            b = temp;
        end
        result = b;
    end
end

% Run the benchmark
run_control_flow_benchmark();