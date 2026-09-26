#[path = "support/mod.rs"]
mod test_helpers;
use test_helpers::execute_source;

#[test]
fn numeric_callable_and_arguments_block_guards_share_exact_predicates() {
    execute_source(r#"
function y = exact_guard(x)
 arguments
  x {mustBeFinite,mustBeInteger,mustBeNonNan,mustBeNonmissing,mustBePositive,mustBeNonzero,mustBeReal}
 end
 y = x;
end
x = uint64(7);
y = exact_guard(x);
mustBeGreaterThan(y,uint64(6)); mustBeGreaterThanOrEqual(y,y);
mustBeLessThan(y,uint64(8)); mustBeLessThanOrEqual(y,y);
mustBeInRange(y,y,y); mustBeMember(y,uint64([3 7]));
mustBeNonnegative(y); mustBeNegative(int64(-7)); mustBeNonpositive(int64(-7));
assert(isa(y,'uint64')); assert(y == x);
"#).unwrap();
}

#[test]
fn arguments_block_missingness_traverses_nested_cells() {
    let error = execute_source(
        r#"
function y = checked(x)
 arguments
  x {mustBeNonmissing}
 end
 y = x;
end
y = checked({{NaN}});
"#,
    )
    .unwrap_err();
    assert!(error.to_string().contains("mustBeNonmissing"), "{error}");
}

#[test]
fn arguments_block_folder_guard_uses_the_shared_filesystem_predicate() {
    execute_source(
        r#"
function y = folder_guard(x)
 arguments
  x {mustBeFolder}
 end
 y = x;
end
directory = tempdir;
checked = folder_guard(directory);
mustBeFolder(checked);
assert(strcmp(checked,directory));
"#,
    )
    .unwrap();
}

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
