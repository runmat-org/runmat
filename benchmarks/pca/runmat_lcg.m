n_default = 200000;
d_default = 1024;
k_default = 8;
iters_default = 15;
seed_default = 0;

env_n = getenv('PCA_N');
if numel(env_n)
  n_default = str2double(env_n);
end
env_d = getenv('PCA_D');
if numel(env_d)
  d_default = str2double(env_d);
end
env_k = getenv('PCA_K');
if numel(env_k)
  k_default = str2double(env_k);
end
env_iters = getenv('PCA_ITERS');
if numel(env_iters)
  iters_default = str2double(env_iters);
end

if ~exist('n','var'), n = n_default; end
if ~exist('d','var'), d = d_default; end
if ~exist('k','var'), k = k_default; end
if ~exist('iters','var'), iters = iters_default; end
if ~exist('seed','var'), seed = seed_default; end

bid = reshape(0:n-1, [n 1]);
cid = reshape(0:d-1, [1 d]);
seed_shift = double(seed);
idx = double(bid) .* double(d) + double(cid) + seed_shift;
state = mod(1664525.0 .* idx + 1013904223.0, 4294967296.0);
A = gpuArray(single(state ./ 4294967296.0));
mu = mean(A, 1);
A = A - mu;
G = (A.' * A) / single(n - 1);
if numel(getenv('RUNMAT_DEBUG_PCA'))
  sum_abs_A = double(gather(sum(abs(A), 'all', 'native')));
  sum_abs_G = double(gather(sum(abs(G), 'all', 'native')));
  fprintf('DEBUG sum_abs_A=%.6e sum_abs_G=%.6e\n', sum_abs_A, sum_abs_G);
end

k_use = min(k, d);
if k_use < 1
  error('PCA requires k >= 1');
end
q_bid = reshape(0:d-1, [d 1]);
q_cid = reshape(0:k_use-1, [1 k_use]);
seed_shift_q = double(seed) + double(n) * double(d);
q_idx = double(q_bid) .* double(k_use) + double(q_cid) + seed_shift_q;
q_state = mod(1664525.0 .* q_idx + 1013904223.0, 4294967296.0);
Q_cpu = single(q_state ./ 4294967296.0);
[Q_cpu, ~] = qr(Q_cpu, 0);
if numel(getenv('RUNMAT_DEBUG_PCA'))
  fprintf('DEBUG max_Q_cpu=%.6e\n', max(abs(Q_cpu), [], 'all'));
end
Q = gpuArray(single(Q_cpu));
if numel(getenv('RUNMAT_DEBUG_PCA'))
  q_gpu_host = gather(abs(Q));
  fprintf('DEBUG max_Q_gpu=%.6e\n', max(q_gpu_host, [], 'all'));
end

for t = 1:iters
  Z = G * Q;
  if numel(getenv('RUNMAT_DEBUG_PCA'))
    nan_Z = gather(sum(isnan(Z), 'all'));
    if nan_Z > 0
      fprintf('DEBUG nan_Z=%d at iter=%d\n', nan_Z, t);
    end
  end
  [Q, ~] = qr(Z, 0);
  Q = single(Q);
  if numel(getenv('RUNMAT_DEBUG_PCA'))
    nan_iter = gather(sum(isnan(Q), 'all'));
    if nan_iter > 0
      fprintf('DEBUG nan_Q_iter=%d at iter=%d\n', nan_iter, t);
    end
  end
end

QtG = Q.' * G;
B = QtG * Q;
Lambda_vec = diag(B);
trace_G_gpu = sum(diag(G), 'all', 'native');

Lambda_host = double(gather(Lambda_vec));
trace_G = double(gather(trace_G_gpu));
explained = Lambda_host / trace_G;

if numel(getenv('RUNMAT_DEBUG_PCA'))
  nan_Q = gather(sum(isnan(Q), 'all'));
  nan_B = gather(sum(isnan(B), 'all'));
  fprintf('DEBUG nan_Q=%d nan_B=%d\n', nan_Q, nan_B);
  preview_len = min(length(Lambda_host), 4);
  fprintf('DEBUG trace_G=%.6e Lambda_top_sum=%.6e lambda_top:', trace_G, sum(Lambda_host));
  for i = 1:preview_len
    fprintf(' %.6e', Lambda_host(i));
  end
  fprintf('\n');
end

fprintf('RESULT_ok EXPLAINED1=%.4f TOPK_SUM=%.6e\n', explained(1), sum(Lambda_host));