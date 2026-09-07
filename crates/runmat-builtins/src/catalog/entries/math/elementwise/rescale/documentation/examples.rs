use crate::{
    BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "unit-interval",
        title: "Scale a vector to the unit interval",
        program: "A = 1:5;\nR = rescale(A)",
        display_output: Some("R = [0 0.25 0.5 0.75 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(R, [0 0.25 0.5 0.75 1]));" },
    },
    BuiltinExample {
        id: "custom-interval",
        title: "Scale a vector to a custom interval",
        program: "A = 1:5;\nR = rescale(A, -1, 1)",
        display_output: Some("R = [-1 -0.5 0 0.5 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(R, [-1 -0.5 0 0.5 1]));" },
    },
    BuiltinExample {
        id: "clip-input-range",
        title: "Clip to an input range before scaling",
        program: "A = [-30 1 2 3 4 5 70];\nR = rescale(A, \"InputMin\", 1, \"InputMax\", 5)",
        display_output: Some("R = [0 0 0.25 0.5 0.75 1 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(R, [0 0 0.25 0.5 0.75 1 1]));" },
    },
    BuiltinExample {
        id: "column-ranges",
        title: "Scale matrix columns independently",
        program: "A = [0.4 -4; 0.5 -5; 0.9 9; 0.2 1];\nR = rescale(A, \"InputMin\", min(A), \"InputMax\", max(A))",
        display_output: Some("R = [0.2857 0.0714; 0.4286 0; 1 1; 0 0.4286]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "expected = [2/7 1/14; 3/7 0; 1 1; 0 3/7];\nassert(max(abs(R(:) - expected(:))) < 1e-12);" },
    },
    BuiltinExample {
        id: "single-storage",
        title: "Preserve single-precision output storage",
        program: "A = single([2 4 6]);\nR = rescale(A)",
        display_output: Some("R = single([0 0.5 1])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(R, \"single\"));\nassert(isequal(R, single([0 0.5 1])));" },
    },
];
