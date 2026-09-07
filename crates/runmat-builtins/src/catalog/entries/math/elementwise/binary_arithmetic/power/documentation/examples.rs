use super::super::super::documentation::{matlab_example, runmat_wgpu_example};
use crate::BuiltinExample;

pub(super) const EXAMPLES: &[BuiltinExample] = &[
    matlab_example(
        "scalar",
        "Raise a scalar to a power",
        "y = power(2, 5)",
        "y = 32",
        "assert(y == 32);",
    ),
    matlab_example(
        "matrix",
        "Square each matrix element",
        "A = [1 2 3; 4 5 6];\nB = power(A, 2)",
        "B = [1 4 9; 16 25 36]",
        "assert(isequal(B, [1 4 9; 16 25 36]));",
    ),
    matlab_example(
        "implicit-expansion",
        "Expand bases and exponents",
        "base = (1:3)';\nexponent = [1 2 3];\nP = power(base, exponent)",
        "P = [1 1 1; 2 4 8; 3 9 27]",
        "assert(isequal(P, [1 1 1; 2 4 8; 3 9 27]));",
    ),
    matlab_example(
        "complex-result",
        "Produce principal complex results",
        "P = power([-2 -1 0 1 2], 0.5)",
        "P = [sqrt(2)i i 0 1 sqrt(2)]",
        "expected = [sqrt(2)*i i 0 1 sqrt(2)]; assert(max(abs(P - expected)) < 1e-12);",
    ),
    matlab_example(
        "characters",
        "Raise character code points",
        "P = power('ABC', 2)",
        "P = [4225 4356 4489]",
        "assert(isequal(P, [4225 4356 4489]));",
    ),
    runmat_wgpu_example(
        "gpu-like",
        "Request provider-resident powers",
        "prototype = gpuArray(single(0));\nP = power(single([1 2 3]), single([2 3 4]), 'like', prototype);\nR = gather(P)",
        "R = single([1 8 81])",
        "assert(isa(P, 'gpuArray')); assert(isequal(R, single([1 8 81])));",
    ),
];
