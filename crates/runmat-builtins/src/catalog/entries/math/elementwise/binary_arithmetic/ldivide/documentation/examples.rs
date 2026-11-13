use super::super::super::documentation::{matlab_example, runmat_wgpu_example};
use crate::BuiltinExample;

pub(super) const EXAMPLES: &[BuiltinExample] = &[
    matlab_example(
        "scalar",
        "Left-divide a vector by a scalar",
        "A = 2;\nB = [4 6 8];\nQ = ldivide(A, B)",
        "Q = [2 3 4]",
        "assert(isequal(Q, [2 3 4]));",
    ),
    matlab_example(
        "implicit-expansion",
        "Expand column divisors and row numerators",
        "A = (1:3)';\nB = [10 20 40];\nQ = ldivide(A, B)",
        "Q = B ./ A",
        "expected = B ./ A; assert(max(abs(Q(:) - expected(:))) < 1e-12);",
    ),
    matlab_example(
        "complex",
        "Left-divide complex values",
        "A = [1+2i, 3-4i];\nB = [2-1i, -1+1i];\nQ = ldivide(A, B)",
        "Q = B ./ A",
        "expected = B ./ A; assert(max(abs(Q - expected)) < 1e-12);",
    ),
    matlab_example(
        "characters",
        "Use character code points as divisors",
        "Q = ldivide('ABC', 2)",
        "Q = 2 ./ [65 66 67]",
        "assert(max(abs(Q - 2 ./ [65 66 67])) < 1e-12);",
    ),
    matlab_example(
        "reciprocal",
        "Compute reciprocals",
        "A = [1 2 4 8];\nQ = ldivide(A, 1)",
        "Q = [1 0.5 0.25 0.125]",
        "assert(isequal(Q, [1 0.5 0.25 0.125]));",
    ),
    runmat_wgpu_example(
        "gpu-like",
        "Request a provider-resident left quotient",
        "prototype = gpuArray(single(0));\nA = gpuArray(single([2 4 8 16]));\nB = gpuArray(single([4 8 16 32]));\nQ = ldivide(A, B, 'like', prototype);\nR = gather(Q)",
        "R = single([2 2 2 2])",
        "assert(isa(Q, 'gpuArray')); assert(isequal(R, single([2 2 2 2])));",
    ),
];
