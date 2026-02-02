mod parse;
use parse::parse;

const PCA_SCRIPT: &str = r"n_default = 200000;
d_default = 1024;
k_default = 8;
iters_default = 15;
seed_default = 0;

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

Q = gpuArray(rand(d, k, 'single'));
[Q, R_unused] = qr(Q, 'econ');

for t = 1:iters
  [Q, R_unused] = qr(G * Q, 'econ');
end

Lambda = diag(Q.' * G * Q);
explained = double(Lambda) / sum(double(diag(G)));
fprintf('RESULT_ok EXPLAINED1=%.4f TOPK_SUM=%.6e\n', explained(1), sum(double(Lambda)));
";

#[test]
fn parse_pca_script() {
    parse("n_default = 1;").expect("simple assignment");
    parse("[Q, R_unused] = qr(G * Q, 'econ');").expect("multi assign call");
    parse("if ~exist('n','var'), n = n_default; end").expect("if with inline assignment");
    let for_block = "for t = 1:iters\n  [Q, R_unused] = qr(G * Q, 'econ');\nend\n";
    parse(for_block).expect("for loop block");
    parse(PCA_SCRIPT).expect("parser should accept PCA script");
}
