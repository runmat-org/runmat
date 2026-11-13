use super::super::ordering_documentation::define_ordering_documentation;

define_ordering_documentation!(
    constant: GT_DOCUMENTATION,
    name: "gt",
    operator: ">",
    relation: "greater than",
    scalar_program: "tf = gt(42, 17)",
    matrix_program: "A = [1 2 3; 4 5 6];\ntf = gt(A, 3)",
    matrix_expected: "[0 0 0; 1 1 1]",
    expansion_program: "tf = gt([1; 3], [2 3 4])",
    expansion_expected: "[0 0 0; 1 0 0]",
    character_program: "tf = gt(['A' 'B' 'C'], 66)",
    character_expected: "[0 0 1]",
    string_program: "tf = gt([\"alice\" \"charlie\" \"bob\"], \"bob\")",
    string_expected: "[0 1 0]",
    gpu_program: "A = gpuArray([1 4 7]);\nB = gpuArray([0 5 6]);\ngtf = gt(A, B);\ntf = gather(gtf)",
    gpu_expected: "[1 0 1]",
    unit_tests: "crates/runmat-runtime/src/builtins/logical/rel/gt/tests.rs",
    wgpu_test: "crates/runmat-runtime/src/builtins/logical/rel/gt/tests.rs::wgpu_matches_host"
);
