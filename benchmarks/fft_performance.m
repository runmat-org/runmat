% FFT Performance Benchmark
% Tests Fast Fourier Transform performance

function run_fft_benchmark()
    fprintf('Running FFT performance benchmark...\n');
    
    % Test different FFT sizes
    sizes = [1024, 8192, 65536];
    
    for i = 1:length(sizes)
        n = sizes(i);
        fprintf('\n=== FFT size: %d points ===\n', n);
        
        % Generate test signal
        t = (0:n-1) / n;
        signal = sin(2*pi*50*t) + 0.5*sin(2*pi*120*t) + 0.1*randn(1,n);
        
        % Forward FFT
        tic;
        Y = fft(signal);
        fft_time = toc;
        fprintf('Forward FFT: %.6f seconds\n', fft_time);
        
        % Inverse FFT
        tic;
        reconstructed = ifft(Y);
        ifft_time = toc;
        fprintf('Inverse FFT: %.6f seconds\n', ifft_time);
        
        % Verify reconstruction quality
        error = mean(abs(signal - real(reconstructed)));
        fprintf('Reconstruction error: %.2e\n', error);
    end
end

% Run the benchmark
run_fft_benchmark();