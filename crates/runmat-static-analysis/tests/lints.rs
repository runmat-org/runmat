use runmat_hir::{LoweringContext, LoweringResult};
use runmat_static_analysis::lints::data_api::lint_data_api_with_provider;
use runmat_static_analysis::schema::{DatasetSchema, DatasetSchemaProvider};

use runmat_runtime as _;

fn lower_result(code: &str) -> LoweringResult {
    let ast = runmat_parser::parse(code).unwrap();
    runmat_hir::lower(&ast, &LoweringContext::empty()).unwrap()
}

#[derive(Default)]
struct MockProvider;

impl DatasetSchemaProvider for MockProvider {
    fn load_schema(&self, dataset_path: &str) -> Option<DatasetSchema> {
        if dataset_path != "/datasets/weather.data" {
            return None;
        }
        let mut arrays = std::collections::HashMap::new();
        arrays.insert("temperature".to_string(), 2usize);
        Some(DatasetSchema { arrays })
    }
}

#[test]
fn data_lint_reports_unknown_array_name() {
    let result =
        lower_result("ds = data.open('/datasets/weather.data'); A = ds.array('humidity');");
    let provider = MockProvider;
    let diags = lint_data_api_with_provider(&result, &provider);
    assert!(diags
        .iter()
        .any(|d| d.code == "lint.data.unknown_array_name"));
}

#[test]
fn data_lint_reports_invalid_slice_rank() {
    let result = lower_result(
        "ds = data.open('/datasets/weather.data'); A = ds.array('temperature'); x = A.read({1,2,3});",
    );
    let provider = MockProvider;
    let diags = lint_data_api_with_provider(&result, &provider);
    assert!(diags
        .iter()
        .any(|d| d.code == "lint.data.invalid_slice_rank"));
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

#[test]
fn data_lint_reports_no_untyped_open() {
    let result = lower_result("runid = 'abc'; p = strcat('/datasets/', runid); ds = data.open(p);");
    let diags = runmat_static_analysis::lint_data_api(&result);
    assert!(diags.iter().any(|d| d.code == "lint.data.no_untyped_open"));
}

#[test]
fn data_lint_allows_literal_open_without_schema() {
    let result = lower_result("ds = data.open('/datasets/weather.data');");
    let diags = runmat_static_analysis::lint_data_api(&result);
    assert!(!diags.iter().any(|d| d.code == "lint.data.no_untyped_open"));
}

#[test]
fn data_lint_reports_multiwrite_outside_tx() {
    let result = lower_result("A = zeros(2,2); A.write(rand(2,2)); A.write(rand(2,2));");
    let diags = runmat_static_analysis::lint_data_api(&result);
    assert!(diags
        .iter()
        .any(|d| d.code == "lint.data.no_multiwrite_outside_tx"));
}

#[test]
fn data_lint_reports_ignore_commit_result_guidance() {
    let result = lower_result("tx = data.open('/datasets/weather.data'); tx.commit();");
    let diags = runmat_static_analysis::lint_data_api(&result);
    assert!(diags
        .iter()
        .any(|d| d.code == "lint.data.ignore_commit_result"));
}

#[test]
fn data_lint_allows_multiwrite_inside_transaction_with_commit_check() {
    let source = "ds = data.open('/datasets/weather.data'); tx = ds.begin(); tx.write('a',{1:2},rand(2,1)); tx.write('b',{1:2},rand(2,1)); ok = tx.commit(); if ~ok; error('failed'); end";
    let result = lower_result(source);
    let diags = runmat_static_analysis::lint_data_api(&result);
    assert!(!diags
        .iter()
        .any(|d| d.code == "lint.data.no_multiwrite_outside_tx"));
    assert!(!diags
        .iter()
        .any(|d| d.code == "lint.data.ignore_commit_result"));
}
