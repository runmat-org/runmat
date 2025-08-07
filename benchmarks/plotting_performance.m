% Plotting Performance Benchmark
% Tests plotting speed and responsiveness

function run_plotting_benchmark()
    fprintf('Running plotting performance benchmark...\n');
    
    % Test different plot sizes
    sizes = [1000, 10000, 50000];
    
    for i = 1:length(sizes)
        n = sizes(i);
        fprintf('\n=== Plot size: %d points ===\n', n);
        
        % Generate test data
        x = linspace(0, 4*pi, n);
        y = sin(x) + 0.1 * randn(1, n);
        
        % Line plot performance
        tic;
        figure('visible', 'off'); % Don't show window for benchmarking
        plot(x, y);
        title(sprintf('Line Plot - %d points', n));
        xlabel('X axis');
        ylabel('Y axis');
        line_time = toc;
        close;
        fprintf('Line plot: %.4f seconds\n', line_time);
        
        % Scatter plot performance
        tic;
        figure('visible', 'off');
        scatter(x(1:min(n,5000)), y(1:min(n,5000))); % Limit scatter points
        title(sprintf('Scatter Plot - %d points', min(n,5000)));
        scatter_time = toc;
        close;
        fprintf('Scatter plot: %.4f seconds\n', scatter_time);
        
        % Bar plot performance (smaller dataset)
        if n <= 1000
            tic;
            figure('visible', 'off');
            bar(x(1:100), y(1:100));
            title('Bar Plot - 100 bars');
            bar_time = toc;
            close;
            fprintf('Bar plot: %.4f seconds\n', bar_time);
        end
    end
end

% Run the benchmark
run_plotting_benchmark();