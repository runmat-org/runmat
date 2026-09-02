use super::super::ordering_documentation::define_ordering_documentation;

define_ordering_documentation!(
    constant: LT_DOCUMENTATION,
    name: "lt",
    operator: "<",
    relation: "less than",
    scalar_program: "tf = lt(17, 42)",
    matrix_program: "A = [1 2 3; 4 5 6];\ntf = lt(A, 3)",
    matrix_expected: "[1 1 0; 0 0 0]",
    expansion_program: "tf = lt([1; 3], [2 3 4])",
    expansion_expected: "[1 1 1; 0 0 1]",
    character_program: "tf = lt(['A' 'B' 'C'], 67)",
    character_expected: "[1 1 0]",
    string_program: "tf = lt([\"alice\" \"charlie\" \"bob\"], \"bob\")",
    string_expected: "[1 0 0]",
    gpu_program: "A = gpuArray([1 4 7]);\nB = gpuArray([2 4 8]);\ngtf = lt(A, B);\ntf = gather(gtf)",
    gpu_expected: "[1 0 1]",
    unit_tests: "crates/runmat-runtime/src/builtins/logical/rel/lt/tests.rs",
    wgpu_test: "crates/runmat-runtime/src/builtins/logical/rel/lt/tests.rs::wgpu_matches_host"
);
