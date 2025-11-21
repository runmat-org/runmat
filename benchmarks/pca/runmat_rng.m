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
rng(seed);

A = gpuArray(rand(n, d, 'single'));
mu = mean(A, 1);
A = A - mu;
G = (A.' * A) / single(n - 1);

G_cpu = double(gather(G));
Lambda_full = eig(G_cpu);
Lambda_sorted = sort(real(Lambda_full), 'descend');
k_use = min(k, length(Lambda_sorted));
Lambda_top = Lambda_sorted(1:k_use);
trace_G = sum(Lambda_full);
explained = double(Lambda_top) / double(trace_G);
fprintf('RESULT_ok EXPLAINED1=%.4f TOPK_SUM=%.6e\n', explained(1), sum(double(Lambda_top)));