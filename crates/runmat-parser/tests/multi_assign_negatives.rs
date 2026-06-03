mod parse;
use parse::parse;

#[test]
fn non_lvalue_in_multi_assign_is_rejected() {
    // Indexed lvalues are valid multi-assign targets; arbitrary expressions are not.
    assert!(parse("[a, x + 1] = f(x)").is_err());
}

#[test]
fn empty_lhs_is_rejected() {
    // '[] = f(x)' is rejected by the grammar (we only allow identifiers or '~')
    assert!(parse("[] = f(x)").is_err());
}

#[test]
fn only_placeholders_are_allowed_syntax() {
    // '[~,~] = f(x)' should parse successfully
    assert!(parse("[~,~] = f(x)").is_ok());
}
