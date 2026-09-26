#[path = "support/mod.rs"]
mod test_helpers;
use test_helpers::execute_source;

#[test]
fn validators_execute_as_statements_without_public_results() {
    execute_source("mustBeNumeric(uint64(7)); mustBeVector([1 2]); mustBeScalarOrEmpty(zeros(2,3,0)); mustBeText('text'); passed = true; assert(passed);").unwrap();
}

#[test]
fn assigned_validator_output_is_rejected() {
    let error = execute_source("result = mustBeNumeric(1);").unwrap_err();
    assert!(error.to_string().contains("output"), "{error}");
}

#[test]
fn arguments_block_sparse_and_nonsparse_checks_use_shared_domain_predicates() {
    execute_source(
        r#"
function y = dense_guard(x)
 arguments
  x {mustBeNonsparse}
 end
 y = x;
end
function y = sparse_guard(x)
 arguments
  x {mustBeSparse}
 end
 y = x;
end
a = dense_guard(uint64(7));
b = sparse_guard(sparse(eye(2)));
assert(isa(a,'uint64')); assert(issparse(b));
"#,
    )
    .unwrap();
}
