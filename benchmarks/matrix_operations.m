% Matrix Operations Benchmark
% Tests basic matrix arithmetic performance

function run_matrix_benchmark()
    fprintf('Running matrix operations benchmark...\n');
    
    % Test matrix sizes
    sizes = [100, 500, 1000];
    operations = {'addition', 'multiplication', 'transpose', 'inverse'};
    
    for i = 1:length(sizes)
        n = sizes(i);
        fprintf('\n=== Matrix size: %dx%d ===\n', n, n);
        
        % Generate test matrices
        A = randn(n, n);
        B = randn(n, n);
        
        % Matrix addition
        tic;
        C = A + B;
        add_time = toc;
        fprintf('Addition: %.4f seconds\n', add_time);
        
        % Matrix multiplication
        tic;
        D = A * B;
        mult_time = toc;
        fprintf('Multiplication: %.4f seconds\n', mult_time);
        
        % Matrix transpose
        tic;
        E = A';
        transpose_time = toc;
        fprintf('Transpose: %.4f seconds\n', transpose_time);
        
        % Matrix inverse (only for smaller matrices)
        if n <= 500
            tic;
            F = inv(A);
            inv_time = toc;
            fprintf('Inverse: %.4f seconds\n', inv_time);
        end
    end
end

% Run the benchmark
run_matrix_benchmark();