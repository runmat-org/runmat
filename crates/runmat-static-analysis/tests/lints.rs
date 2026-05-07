use runmat_hir::{LoweringContext, LoweringResult};

use runmat_runtime as _;

fn lower_result(code: &str) -> LoweringResult {
    let ast = runmat_parser::parse(code).unwrap();
    runmat_hir::lower(&ast, &LoweringContext::empty()).unwrap()
}

#[test]
fn shape_lint_reports_matmul_mismatch() {
    let result = lower_result("a = ones(2,3); b = ones(4,2); c = a * b;");
    let diags = runmat_static_analysis::lint_shapes(&result);
    assert!(diags.iter().any(|d| d.code == "lint.shape.matmul"));
}

#[test]
fn shape_lint_reports_broadcast_mismatch() {
    let result = lower_result("a = ones(2,3); b = ones(4,2); c = a + b;");
    let diags = runmat_static_analysis::lint_shapes(&result);
    assert!(diags.iter().any(|d| d.code == "lint.shape.broadcast"));
}

#[test]
fn shape_lint_reports_dot_and_reshape() {
    let result = lower_result(
        "a = ones(1,3); b = ones(1,4); c = dot(a, b); d = reshape(a, 2, 2); e = reshape(a, -1, -1);",
    );
    let diags = runmat_static_analysis::lint_shapes(&result);
    assert!(diags.iter().any(|d| d.code == "lint.shape.dot"));
    assert!(diags.iter().any(|d| d.code == "lint.shape.reshape"));
}

#[test]
fn shape_lint_reports_logical_index_mismatch() {
    let result = lower_result("a = ones(2,2); m = ones(1,2) > 0; b = a[m];");
    let diags = runmat_static_analysis::lint_shapes(&result);
    assert!(diags.iter().any(|d| d.code == "lint.shape.logical_index"));
}

#[test]
fn shape_lint_reports_repmat_and_permute() {
    let bad_result = lower_result(
        "a = ones(2,2); b = repmat(a, 1.5, 2); c = permute(a, [1 2 3]); d = permute(a, [1 1]);",
    );
    let bad_diags = runmat_static_analysis::lint_shapes(&bad_result);
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.repmat"));
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.permute"));

    let good_result = lower_result("a = ones(2,2); b = repmat(a, 2, 3); c = permute(a, [2 1]);");
    let good_diags = runmat_static_analysis::lint_shapes(&good_result);
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.repmat"));
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.permute"));
}

#[test]
fn shape_lint_reports_concat_mismatches() {
    let bad_result =
        lower_result("B = ones(2,3); C = ones(4,3); D = ones(2,4); A = [B, C]; E = [B; D];");
    let bad_diags = runmat_static_analysis::lint_shapes(&bad_result);
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.horzcat"));
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.vertcat"));

    let good_result =
        lower_result("B = ones(2,3); C = ones(2,4); D = ones(4,3); A = [B, C]; E = [B; D];");
    let good_diags = runmat_static_analysis::lint_shapes(&good_result);
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.horzcat"));
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.vertcat"));
}

#[test]
fn shape_lint_reports_reduction_dim_out_of_range() {
    let bad_result = lower_result("a = ones(2,2); b = sum(a, 3);");
    let bad_diags = runmat_static_analysis::lint_shapes(&bad_result);
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.reduction"));

    let good_result = lower_result("a = ones(2,2); b = sum(a, 2);");
    let good_diags = runmat_static_analysis::lint_shapes(&good_result);
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.reduction"));
}
