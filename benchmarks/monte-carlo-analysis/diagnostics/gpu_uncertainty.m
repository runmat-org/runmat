% Official v0.6.2: default fused reductions give stderr ~2294.188.
% CPU-only reference: price ~7.25485852, stderr ~0.02743129856.
% RUNMAT_DISABLE_FUSED_REDUCTION=1 makes this input pass.
x = single(sin((1:100000)'));
g = gpuArray(x);
p = exp(-0.03) .* max(100 .* exp(0.01 + 0.2 .* g) - 100, 0);
a = mean(p);
b = std(p) / sqrt(numel(x));
a = gather(a);
b = gather(b);
fprintf('gpu price=%.12g stderr=%.12g\n', a, b);
