use super::super::ordering_documentation::define_ordering_documentation;

define_ordering_documentation!(
    constant: LE_DOCUMENTATION,
    name: "le",
    operator: "<=",
    relation: "less than or equal to",
    scalar_program: "tf = le(17, 42)",
    matrix_program: "A = [1 2 3; 4 5 6];\ntf = le(A, 3)",
    matrix_expected: "[1 1 1; 0 0 0]",
    expansion_program: "tf = le([1; 3], [2 3 4])",
    expansion_expected: "[1 1 1; 0 1 1]",
    character_program: "tf = le(['A' 'B' 'C'], 66)",
    character_expected: "[1 1 0]",
    string_program: "tf = le([\"alice\" \"charlie\" \"bob\"], \"bob\")",
    string_expected: "[1 0 1]",
    gpu_program: "A = gpuArray([1 4 7]);\nB = gpuArray([2 4 8]);\ngtf = le(A, B);\ntf = gather(gtf)",
    gpu_expected: "[1 1 1]",
    unit_tests: "crates/runmat-runtime/src/builtins/logical/rel/le/tests.rs",
    wgpu_test: "crates/runmat-runtime/src/builtins/logical/rel/le/tests.rs::wgpu_matches_host"
);
