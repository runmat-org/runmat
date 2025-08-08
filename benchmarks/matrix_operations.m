function run_matrix_benchmark()
    fprintf('=== Matrix Operations Benchmark ===\n');
    
    sizes = [100, 300, 500];
    for s = 1:length(sizes)
        n = sizes(s);
        fprintf('\nMatrix size: %d\n', n);
        
        A = randn(n, n);
        B = randn(n, n);
        
        tic;
        C = A + B;
        add_time = toc;
        fprintf('  Addition time: %.6f\n', add_time);
        
        tic;
        D = A * B;
        mult_time = toc;
        fprintf('  Multiplication time: %.6f\n', mult_time);
        
        tic;
        E = A';
        transpose_time = toc;
        fprintf('  Transpose time: %.6f\n', transpose_time);
        
        tic;
        F = A * 2.0;
        scalar_time = toc;
        fprintf('  Scalar multiplication time: %.6f\n', scalar_time);
    end
    
    fprintf('\nMatrix benchmark completed.\n');
end

run_matrix_benchmark();