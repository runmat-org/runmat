#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_numeric_limits_preserve_selected_classes_and_like_representations() {
    execute_source(
        "default_integer = intmax(); \
         if ~isa(default_integer, 'int32') || default_integer ~= intmax('int32'); error('default integer limit'); end; \
         wide = intmax('uint64'); \
         if ~isa(wide, 'uint64') || wide ~= uint64(18446744073709551615); error('wide integer limit'); end; \
         minimum = intmin('int64'); \
         if ~isa(minimum, 'int64') || minimum ~= int64(-9223372036854775808); error('signed integer limit'); end; \
         single_limit = realmax('single'); \
         if ~isa(single_limit, 'single'); error('single floating limit'); end; \
         complex_prototype = complex(single(1), single(2)); \
         complex_limit = flintmax('like', complex_prototype); \
         if ~isa(complex_limit, 'single') || isreal(complex_limit); error('complex floating limit'); end; \
         sparse_prototype = sparse(complex_prototype); \
         sparse_limit = realmin('like', sparse_prototype); \
         if ~isa(sparse_limit, 'single'); error('sparse floating class'); end; \
         if ~issparse(sparse_limit); error('sparse floating storage'); end; \
         if isreal(sparse_limit); error('sparse floating complexity'); end;",
    )
    .expect("compiled numeric-limit contracts");
}

#[test]
fn compiled_numeric_limits_report_canonical_contract_errors() {
    let error = execute_source("x = intmax('double');")
        .expect_err("integer limits reject floating classes");
    assert_eq!(
        error.identifier(),
        Some("RunMat:numericLimits:InvalidClass")
    );

    let error = execute_source("x = realmin('like', uint8(1));")
        .expect_err("floating limits reject integer prototypes");
    assert_eq!(
        error.identifier(),
        Some("RunMat:numericLimits:InvalidClass")
    );
}
