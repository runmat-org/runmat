% Linear Algebra Performance Benchmark
% Tests advanced linear algebra operations

function run_linalg_benchmark()
    fprintf('Running linear algebra benchmark...\n');
    
    % Test matrix sizes
    sizes = [200, 500, 1000];
    
    for i = 1:length(sizes)
        n = sizes(i);
        fprintf('\n=== Matrix size: %dx%d ===\n', n, n);
        
        % Generate test matrices
        A = randn(n, n);
        B = randn(n, n);
        
        % Eigenvalue decomposition
        tic;
        eigenvals = eig(A);
        eig_time = toc;
        fprintf('Eigenvalues: %.4f seconds\n', eig_time);
        
        % SVD decomposition
        tic;
        [U, S, V] = svd(A);
        svd_time = toc;
        fprintf('SVD decomposition: %.4f seconds\n', svd_time);
        
        % QR decomposition
        tic;
        [Q, R] = qr(A);
        qr_time = toc;
        fprintf('QR decomposition: %.4f seconds\n', qr_time);
        
        % LU decomposition
        tic;
        [L, U, P] = lu(A);
        lu_time = toc;
        fprintf('LU decomposition: %.4f seconds\n', lu_time);
        
        % Matrix solve (Ax = b)
        b = randn(n, 1);
        tic;
        x = A \ b;
        solve_time = toc;
        fprintf('Linear solve: %.4f seconds\n', solve_time);
        
        % Verify solution quality
        residual = norm(A*x - b) / norm(b);
        fprintf('Solve residual: %.2e\n', residual);
    end
end

% Run the benchmark
run_linalg_benchmark();