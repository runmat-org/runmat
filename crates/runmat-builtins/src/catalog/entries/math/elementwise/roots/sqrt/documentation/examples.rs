use crate::{
    BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

pub(super) const SQRT_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "single-pages",
        title: "Preserve single precision and array pages",
        program: "input = reshape(single([4 9]), [1 1 2]);\nresult = sqrt(input);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(result), 'single'));\nassert(isequal(size(result), [1 1 2]));\nassert(max(abs(double(result(:)) - [2; 3])) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "positive-scalar",
        title: "Take the square root of a positive scalar",
        program: "y = sqrt(9)",
        display_output: Some("y = 3"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 3);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Take square roots elementwise across a matrix",
        program: "A = [1 4 9; 16 25 36];\nR = sqrt(A)",
        display_output: Some("R = [1 2 3; 4 5 6]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [1 2 3; 4 5 6]));",
        },
    },
    BuiltinExample {
        id: "negative-real-input",
        title: "Promote negative real input to a complex result",
        program: "values = [-1 -4 9];\nroots = sqrt(values)",
        display_output: Some("roots = [0.0000 + 1.0000i, 0.0000 + 2.0000i, 3.0000 + 0.0000i]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(roots - [1i 2i 3])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Take square roots of provider-resident data",
        program: "G = gpuArray([0 1; 4 9]);\nout = sqrt(G);\nresult = gather(out)",
        display_output: Some("result = [0 1; 2 3]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [0 1; 2 3]));",
        },
    },
    BuiltinExample {
        id: "complex-values",
        title: "Take principal square roots of complex values",
        program: "z = [3 + 4i, -1 + 2i];\nw = sqrt(z)",
        display_output: Some("w = [2 + 1i, 0.7862 + 1.2720i]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(w .* w - z)) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Use the numeric code points of a character array",
        program: "C = 'AB';\nnumericRoots = sqrt(C)",
        display_output: Some("numericRoots = [8.0623 8.2462]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(numericRoots - sqrt([65 66]))) < 1e-12);",
        },
    },
];
