use super::super::ordering_documentation::define_ordering_documentation;

define_ordering_documentation!(
    constant: GE_DOCUMENTATION,
    name: "ge",
    operator: ">=",
    relation: "greater than or equal to",
    scalar_program: "tf = ge(42, 42)",
    matrix_program: "A = [1 2 3; 4 5 6];\ntf = ge(A, 3)",
    matrix_expected: "[0 0 1; 1 1 1]",
    expansion_program: "tf = ge([1; 3], [2 3 4])",
    expansion_expected: "[0 0 0; 1 1 0]",
    character_program: "tf = ge(['A' 'B' 'C'], 66)",
    character_expected: "[0 1 1]",
    string_program: "tf = ge([\"alice\" \"charlie\" \"bob\"], \"bob\")",
    string_expected: "[0 1 1]",
    gpu_program: "A = gpuArray([1 4 6]);\nB = gpuArray([1 5 6]);\ngtf = ge(A, B);\ntf = gather(gtf)",
    gpu_expected: "[1 0 1]",
    unit_tests: "crates/runmat-runtime/src/builtins/logical/rel/ge/tests.rs",
    wgpu_test: "crates/runmat-runtime/src/builtins/logical/rel/ge/tests.rs::wgpu_matches_host"
);
