#[test]
fn seed_script_with_matrix_rows_parses() {
    let src = "seed = 0;\n    G = [\n        2, -1, 0;\n        0, 1, 3;\n        4, 2, -2\n    ];\n    Q = [\n        1, 2;\n        3, 4;\n        5, 6\n    ];\n    Q = mtimes(G, Q);\n    norms = sqrt(sum(Q.^2, 1) + 1e-6);\n    Q = Q ./ norms;\n";
    assert!(
        runmat_parser::parse(src).is_ok(),
        "parser should accept matrix rows with newlines"
    );
}
