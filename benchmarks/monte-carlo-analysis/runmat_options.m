% Exact terminal GBM, shared little-endian float64 fixture. Also runs in Octave.
fid = fopen(getenv('MC_FIXTURE'), 'rb', 'ieee-le');
z = fread(fid, Inf, 'double');
fclose(fid);
is_single = strcmp(getenv('MC_PRECISION'), 'float32');
is_gpu = strcmp(getenv('MC_DEVICE'), 'gpu');
roundtrip = strcmp(getenv('MC_TIMING'), 'end-to-end');
if is_single
    z = single(z);
end
host_z = z;
if is_gpu
    device = gpuDevice();
    disp('DEVICE gpu');
    disp(device.name);
    if ~roundtrip
        z = gpuArray(host_z);
        % Force completion of the initial upload outside the timed region.
        upload_check = gather(sum(z));
    end
end
M = numel(z);
mode = getenv('MC_MODE');
repeats = str2double(getenv('MC_REPEATS'));
warmups = str2double(getenv('MC_WARMUPS'));
for rep = 1:(warmups + repeats)
    tic;
    if is_gpu && roundtrip
        z = gpuArray(host_z);
    end
    if strcmp(mode, 'loop')
        payoff = zeros(M, 1);
        for i = 1:M
            payoff(i) = exp(-0.03) * max(100 * exp(0.01 + 0.2*z(i)) - 100, 0);
        end
    else
        payoff = exp(-0.03) .* max(100 .* exp(0.01 + 0.2 .* z) - 100, 0);
    end
    price = mean(payoff);
    stderr_value = std(payoff) / sqrt(M);
    if is_gpu
        price = gather(price);
        stderr_value = gather(stderr_value);
    end
    elapsed_ms = toc * 1000;
    fprintf('SAMPLE rep=%d ms=%.12g price=%.17g stderr=%.17g\n', rep, elapsed_ms, price, stderr_value);
end
