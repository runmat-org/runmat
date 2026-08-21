use runmat_hir::{HirDiagnostic, LoweringContext};

use runmat_runtime as _;

fn diagnostics(code: &str) -> Vec<HirDiagnostic> {
    runmat_static_analysis::frontend::analyze_source(
        code,
        runmat_parser::CompatMode::default(),
        &LoweringContext::empty(),
    )
    .diagnostics
}

#[test]
fn shape_lint_reports_matmul_mismatch() {
    let diags = diagnostics("a = ones(2,3); b = ones(4,2); c = a * b;");
    assert!(diags.iter().any(|d| d.code == "RM-TYPE-MATMUL"));
}

#[test]
fn shape_lint_reports_broadcast_mismatch() {
    let diags = diagnostics("a = ones(2,3); b = ones(4,2); c = a + b;");
    assert!(diags.iter().any(|d| d.code == "RM-TYPE-BROADCAST"));
}

#[test]
fn shape_lint_reports_dot_and_reshape() {
    let diags = diagnostics(
        "a = ones(1,3); b = ones(1,4); c = dot(a, b); d = reshape(a, 2, 2); e = reshape(a, -1, -1);",
    );
    assert!(diags.iter().any(|d| d.code == "RM-TYPE-DOT"));
    assert!(diags.iter().any(|d| d.code == "RM-TYPE-RESHAPE"));
}

#[test]
fn shape_lint_reports_logical_index_mismatch() {
    let diags = diagnostics("a = ones(2,2); m = ones(1,2) > 0; b = a[m];");
    assert!(diags.iter().any(|d| d.code == "RM-TYPE-LOGICAL-INDEX"));
}

#[test]
fn shape_lint_allows_numeric_range_indexing() {
    let diags = diagnostics("a = 0:pi/100:2*pi; b = sin(a); c = a(1:10);");
    assert!(!diags.iter().any(|d| d.code == "RM-TYPE-LOGICAL-INDEX"));
}

#[test]
fn shape_lint_allows_numeric_vector_and_scalar_indexing() {
    let vector_diags = diagnostics("a = ones(1,10); idx = [1 3 5 7]; b = a(idx);");
    assert!(!vector_diags
        .iter()
        .any(|d| d.code == "RM-TYPE-LOGICAL-INDEX"));

    let scalar_diags = diagnostics("a = ones(1,10); b = a(3);");
    assert!(!scalar_diags
        .iter()
        .any(|d| d.code == "RM-TYPE-LOGICAL-INDEX"));
}

#[test]
fn shape_lint_allows_matching_logical_indexing() {
    let diags = diagnostics("a = ones(2,2); m = ones(2,2) > 0; b = a(m);");
    assert!(!diags.iter().any(|d| d.code == "RM-TYPE-LOGICAL-INDEX"));
}

#[test]
fn shape_lint_reports_repmat_and_permute() {
    let bad_diags = diagnostics(
        "a = ones(2,2); b = repmat(a, 1.5, 2); c = permute(a, [1 2 3]); d = permute(a, [1 1]);",
    );
    assert!(bad_diags.iter().any(|d| d.code == "RM-TYPE-REPMAT"));
    assert!(bad_diags.iter().any(|d| d.code == "RM-TYPE-PERMUTE"));

    let good_diags = diagnostics("a = ones(2,2); b = repmat(a, 2, 3); c = permute(a, [2 1]);");
    assert!(!good_diags.iter().any(|d| d.code == "RM-TYPE-REPMAT"));
    assert!(
        !good_diags.iter().any(|d| d.code == "RM-TYPE-PERMUTE"),
        "valid permute produced diagnostics: {good_diags:#?}"
    );
}

#[test]
fn shape_lint_reports_concat_mismatches() {
    let bad_diags =
        diagnostics("B = ones(2,3); C = ones(4,3); D = ones(2,4); A = [B, C]; E = [B; D];");
    assert!(bad_diags.iter().any(|d| d.code == "RM-TYPE-CONCAT"));

    let good_diags =
        diagnostics("B = ones(2,3); C = ones(2,4); D = ones(4,3); A = [B, C]; E = [B; D];");
    assert!(!good_diags.iter().any(|d| d.code == "RM-TYPE-CONCAT"));
}

#[test]
fn shape_lint_reports_reduction_dim_out_of_range() {
    // MATLAB permits dimensions beyond ndims and returns the input unchanged.
    let extended_dim = diagnostics("a = ones(2,2); b = sum(a, 3);");
    assert!(!extended_dim
        .iter()
        .any(|d| d.code == "RM-TYPE-REDUCTION-DIMENSION"));

    let invalid_zero = diagnostics("a = ones(2,2); b = sum(a, 0);");
    assert!(invalid_zero
        .iter()
        .any(|d| d.code == "RM-TYPE-REDUCTION-DIMENSION"));
}
