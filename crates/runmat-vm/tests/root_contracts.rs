#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_roots_preserve_class_shape_and_domain_contracts() {
    execute_source(
        "single_root = sqrt(single([1 4 9])); \
         if ~isa(single_root, 'single') || ~isequal(size(single_root), [1 3]); error('single root'); end; \
         complex_root = sqrt(-4); \
         if isreal(complex_root) || abs(complex_root - 2i) > 1e-12; error('principal root'); end; \
         sparse_root = realsqrt(sparse(single([0 4; 9 0]))); \
         if ~isa(sparse_root, 'single') || ~issparse(sparse_root); error('sparse real root'); end;",
    )
    .expect("compiled root contracts");
}

#[test]
fn compiled_realsqrt_reports_canonical_class_and_domain_errors() {
    let class_error =
        execute_source("x = realsqrt(uint16(4));").expect_err("realsqrt must reject integer input");
    assert_eq!(
        class_error.identifier(),
        Some("RunMat:realsqrt:InvalidInput")
    );

    let domain_error =
        execute_source("x = realsqrt(-4);").expect_err("realsqrt must reject negative input");
    assert_eq!(
        domain_error.identifier(),
        Some("RunMat:realsqrt:ComplexResult")
    );
}
