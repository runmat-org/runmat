use runmat_parser::parse_simple as parse;

#[test]
fn mixed_lvalue_in_multi_assign_is_rejected() {
    // '[a, x(1)] = f(x)' should not parse as a valid multi-assign
    // Parser should fail rather than accept mixed lvalues
    assert!(parse("[a, x(1)] = f(x)").is_err());
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
