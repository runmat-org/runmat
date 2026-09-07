use super::super::super::documentation::{
    matlab_example, matlab_wgpu_example, runmat_wgpu_example,
};
use crate::BuiltinExample;

pub(super) const EXAMPLES: &[BuiltinExample] = &[
    matlab_example(
        "matrices",
        "Divide matrices element by element",
        "A = [8 12 18; 2 10 18];\nB = [2 3 6; 2 5 9];\nQ = rdivide(A, B)",
        "Q = [4 4 3; 1 2 2]",
        "assert(isequal(Q, [4 4 3; 1 2 2]));",
    ),
    matlab_example(
        "scalar",
        "Divide an array by a scalar",
        "A = [9 1 7; 3 10 18; 16 13 4];\nQ = rdivide(A, 2)",
        "Q = [4.5 0.5 3.5; 1.5 5 9; 8 6.5 2]",
        "assert(isequal(Q, A ./ 2));",
    ),
    matlab_example(
        "implicit-expansion",
        "Expand a column and row",
        "col = (1:3)';\nrow = [10 20 30];\nQ = rdivide(col, row)",
        "Q = [0.1 0.05 1/30; 0.2 0.1 1/15; 0.3 0.15 0.1]",
        "expected = col ./ row; assert(max(abs(Q(:) - expected(:))) < 1e-12);",
    ),
    matlab_example(
        "complex",
        "Divide complex values",
        "z1 = [1+2i, 3-4i];\nz2 = [2-1i, -1+1i];\nQ = rdivide(z1, z2)",
        "Q = [1i -3.5+0.5i]",
        "assert(max(abs(Q - [1i -3.5+0.5i])) < 1e-12);",
    ),
    matlab_example(
        "characters",
        "Divide character code points",
        "Q = rdivide('ABC', 2)",
        "Q = [32.5 33 33.5]",
        "assert(isequal(Q, [32.5 33 33.5]));",
    ),
    matlab_wgpu_example(
        "resident",
        "Divide provider-resident arrays",
        "A = gpuArray(single([10 20 30]));\nB = gpuArray(single([2 5 10]));\nQ = rdivide(A, B);\nR = gather(Q)",
        "R = single([5 4 3])",
        "assert(isa(Q, 'gpuArray')); assert(isequal(R, single([5 4 3])));",
    ),
    runmat_wgpu_example(
        "gpu-like",
        "Request a provider-resident quotient",
        "prototype = gpuArray(single(0));\nQ = rdivide(single([1 2 3]), single([2 4 6]), 'like', prototype);\nR = gather(Q)",
        "R = single([0.5 0.5 0.5])",
        "assert(isa(Q, 'gpuArray')); assert(isequal(R, single([0.5 0.5 0.5])));",
    ),
];
