reduce_len = 4096;
len_env = getenv('REDUCE_SWEEP_LEN');
if numel(len_env)
  reduce_len = max(1, round(str2double(len_env)));
end

num_slices = 16;
slices_env = getenv('REDUCE_SWEEP_SLICES');
if numel(slices_env)
  num_slices = max(1, round(str2double(slices_env)));
end

seed = 0;
seed_env = getenv('REDUCE_SWEEP_SEED');
if numel(seed_env)
  seed = round(str2double(seed_env));
end

data = build_lcg_matrix(reduce_len, num_slices, seed);

mean_gpu = mean(data, 1, 'native');
sum_gpu = sum(data, 1, 'native');

data_host = double(gather(data));
mean_ref = mean(data_host, 1);
sum_ref = sum(data_host, 1);

mean_err = max(abs(double(gather(mean_gpu)) - mean_ref), [], 'all');
sum_err = max(abs(double(gather(sum_gpu)) - sum_ref), [], 'all');
max_err = max(mean_err, sum_err);

fprintf('RESULT_ok MAX_ERR=%.6e LEN=%d SLICES=%d MEAN_ERR=%.6e SUM_ERR=%.6e\n', max_err, reduce_len, num_slices, mean_err, sum_err);

function out = build_lcg_matrix(rows, cols, seed)
  out = zeros(rows, cols, 'single');
  rid = double(reshape(0:rows-1, [rows 1]));
  stride = double(rows);
  base = double(seed);
  for c = 1:cols
    idx = rid + stride * double(c - 1) + base;
    state = mod(1664525.0 .* idx + 1013904223.0, 4294967296.0);
    out(:, c) = single(state ./ 4294967296.0);
  end
end

